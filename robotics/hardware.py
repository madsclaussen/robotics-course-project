"""Dynamixel interface for the course robot (Annex C, Python).

Joint angles here use the same convention as the kinematics (radians,
q = 0 straight up). The mapping to servo ticks is
    ticks = center + sign * (q + offset) * ticks_per_rad
and MUST be calibrated on your robot: move it to the Figure 1 pose
(all q = 0), read the raw ticks with `read_raw()`, and check the sign of each
joint by commanding small angles.

Defaults assume AX-12A style servos (0..1023 over 300 deg, centre 512),
which matches the compliance-margin/slope registers used in Annex C.
"""
import os
import termios
import time

import numpy as np

try:
    import dynamixel_sdk as dxl
except ImportError:  # allow importing the module without the SDK (e.g. simulation)
    dxl = None

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_CW_COMPLIANCE_MARGIN = 26
ADDR_MX_CCW_COMPLIANCE_MARGIN = 27
ADDR_MX_CW_COMPLIANCE_SLOPE = 28
ADDR_MX_CCW_COMPLIANCE_SLOPE = 29
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32
ADDR_MX_PRESENT_POSITION = 36
ADDR_MX_MOVING = 46
ADDR_MX_PUNCH = 48
PROTOCOL_VERSION = 1.0
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


class Robot:
    def __init__(self, device="/dev/ttyACM0", baudrate=1_000_000, ids=(1, 2, 3, 4),
                 center=512, ticks_per_rad=1023 / np.deg2rad(300),
                 signs=(1, 1, 1, 1), offsets=(0, 0, 0, 0),
                 q_min=None, q_max=None, speed=100,
                 retries=3, reconnect_timeout=10.0):
        if dxl is None:
            raise ImportError("dynamixel_sdk not installed: pip install dynamixel-sdk")
        self.ids = list(ids)
        self.center = center
        self.ticks_per_rad = ticks_per_rad
        self.signs = np.asarray(signs, float)
        self.offsets = np.asarray(offsets, float)
        self.q_min = None if q_min is None else np.asarray(q_min, float)
        self.q_max = None if q_max is None else np.asarray(q_max, float)
        self.max_tick = 1023 if center == 512 else 4095

        self.device = device
        self.baudrate = baudrate
        self.retries = retries
        self.reconnect_timeout = reconnect_timeout
        self.packet = dxl.PacketHandler(PROTOCOL_VERSION)
        self._open()

        for i in self.ids:
            self._w1(i, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
            self._w1(i, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
            self._w1(i, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
            self._w1(i, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
            self._w1(i, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
        self.set_speed(speed)

    # ---- connection ------------------------------------------------------
    def _open(self):
        self.port = dxl.PortHandler(self.device)
        if not self.port.openPort():
            raise IOError(f"Could not open port {self.device}")
        if not self.port.setBaudRate(self.baudrate):
            raise IOError(f"Could not set baudrate {self.baudrate}")

    def _reconnect(self):
        """The USB adapter dropped off the bus (hub reset). Wait for the device
        node to come back and reopen it. The servos keep power, torque and
        their goal position meanwhile, so the interrupted command can be retried."""
        print(f"warning: lost {self.device}, reconnecting...", flush=True)
        try:
            self.port.closePort()
        except Exception:
            pass
        t0 = time.time()
        last = None
        while time.time() - t0 < self.reconnect_timeout:
            if os.path.exists(self.device):
                try:
                    self._open()
                    print(f"reconnected after {time.time() - t0:.1f} s", flush=True)
                    return
                except Exception as e:  # node exists but not ready / briefly busy
                    last = e
            time.sleep(0.2)
        raise IOError(f"{self.device} did not come back within "
                      f"{self.reconnect_timeout} s (last error: {last})")

    def _txrx(self, fn, *args):
        """Run one SDK TxRx call, reconnecting and retrying if the port vanished."""
        for attempt in range(self.retries + 1):
            try:
                out = fn(self.port, *args)
            except (OSError, termios.error):  # SerialException is an OSError
                if attempt == self.retries:
                    raise
                self.port.is_using = False
                self._reconnect()
                continue
            if out[-2] == dxl.COMM_PORT_BUSY and attempt < self.retries:
                self.port.is_using = False  # stale flag from an earlier exception
                continue
            return out
        return out

    # ---- low level -------------------------------------------------------
    def _check(self, res, err, what):
        if res != dxl.COMM_SUCCESS:
            raise IOError(f"{what}: {self.packet.getTxRxResult(res)}")
        if err != 0:
            print(f"warning {what}: {self.packet.getRxPacketError(err)}")

    def _w1(self, i, addr, val):
        self._check(*self._txrx(self.packet.write1ByteTxRx, i, addr, int(val)), f"id {i} @{addr}")

    def _w2(self, i, addr, val):
        self._check(*self._txrx(self.packet.write2ByteTxRx, i, addr, int(val)), f"id {i} @{addr}")

    def _r2(self, i, addr):
        val, res, err = self._txrx(self.packet.read2ByteTxRx, i, addr)
        self._check(res, err, f"id {i} @{addr}")
        return val

    def _r1(self, i, addr):
        val, res, err = self._txrx(self.packet.read1ByteTxRx, i, addr)
        self._check(res, err, f"id {i} @{addr}")
        return val

    # ---- conversions -----------------------------------------------------
    def q_to_ticks(self, q):
        t = self.center + self.signs * (np.asarray(q) + self.offsets) * self.ticks_per_rad
        return np.clip(np.round(t), 0, self.max_tick).astype(int)

    def ticks_to_q(self, ticks):
        return (np.asarray(ticks) - self.center) / (self.signs * self.ticks_per_rad) - self.offsets

    # ---- high level ------------------------------------------------------
    def set_speed(self, speed):
        """Moving speed register (0..1023, 0 = max speed!)."""
        for i in self.ids:
            self._w2(i, ADDR_MX_MOVING_SPEED, speed)

    def get_speed(self):
            """Read moving speed register (0..1023, 0 = max speed!)."""
            return np.array([self._r2(i, ADDR_MX_MOVING_SPEED) for i in self.ids])    
            

    def read_raw(self):
        return np.array([self._r2(i, ADDR_MX_PRESENT_POSITION) for i in self.ids])

    def read_q(self):
        return self.ticks_to_q(self.read_raw())

    def is_moving(self):
        return any(self._r1(i, ADDR_MX_MOVING) for i in self.ids)

    def move_q(self, q, wait=True, timeout=10.0):
        q = np.asarray(q, float)
        if np.any(np.isnan(q)):
            raise ValueError(f"NaN in joint command {q}")
        if self.q_min is not None and np.any(q < self.q_min):
            raise ValueError(f"joint command {q} below limits {self.q_min}")
        if self.q_max is not None and np.any(q > self.q_max):
            raise ValueError(f"joint command {q} above limits {self.q_max}")
        
        for i, t in zip(self.ids, self.q_to_ticks(q)):
            self._w2(i, ADDR_MX_GOAL_POSITION, t)
        if wait:
            self.wait(timeout)

    def wait(self, timeout=10.0):
        t0 = time.time()
        time.sleep(0.05)
        while self.is_moving() and time.time() - t0 < timeout:
            time.sleep(0.02)

    def torque(self, on=True):
        for i in self.ids:
            self._w1(i, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE if on else TORQUE_DISABLE)

    def close(self, disable_torque=True):
        # A previous failed TxRx (e.g. USB dropped -> EIO) can leave the SDK's
        # busy flag set, which makes every later call fail with "Port is in use!".
        self.port.is_using = False
        try:
            if disable_torque:
                self.torque(False)
        except Exception as e:
            print(f"warning: could not disable torque on close: {e}")
        finally:
            self.port.closePort()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SimRobot:
    """Drop-in stand-in for Robot (no hardware) — useful for testing scripts."""

    def __init__(self, *a, **kw):
        self.q = np.zeros(4)

    def set_speed(self, speed):
        pass

    def read_q(self):
        return self.q.copy()

    def move_q(self, q, wait=True, timeout=10.0):
        self.q = np.asarray(q, float)

    def wait(self, timeout=10.0):
        pass

    def torque(self, on=True):
        pass

    def close(self, disable_torque=True):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
