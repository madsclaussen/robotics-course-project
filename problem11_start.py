"""Problem 11 starting point: connect to the robot + camera and exercise them.

    python problem11_start.py --sim               # no hardware, just runs the logic
    python problem11_start.py --port /dev/ttyACM0 --cam 0

Steps it performs:
  1. connects (and prints the raw servo ticks -> use them to calibrate
     `signs`/`offsets` in CALIB below so that q = 0 is the Figure-1 pose),
  2. moves to a safe home pose,
  3. traces the Problem 3 circle point by point using the Python IK,
  4. shows the camera image with a pixel -> table-plane back-projection
     of the clicked point (press q to quit).
"""
import argparse

import numpy as np

from robotics import circle_points, ik_path, inverse_kinematics, is_reachable, T04
from robotics.hardware import Robot, SimRobot

# ---- calibrate these on your robot ---------------------------------------
CALIB = dict(
    signs=(1, 1, 1, 1),          # flip a sign if the joint moves the wrong way
    offsets=(0, 0, 0, 0),        # [rad] so that commanded q=0 is the Figure-1 pose
    q_min=np.deg2rad([-150, -100, -150, -150]),
    q_max=np.deg2rad([150, 100, 150, 150]),
    speed=80,
)
TABLE_Z = 0.0                    # [mm] height of the table surface in frame 0
# Camera intrinsics (placeholder -- get real values with cv2.calibrateCamera)
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
# ---------------------------------------------------------------------------


def move_to_point(robot, p, x4z=0.0):
    if not is_reachable(p, x4z):
        raise ValueError(f"point {p} not reachable")
    q = inverse_kinematics(p, x4z)
    robot.move_q(q)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--cam", type=int, default=None, help="camera index (omit to skip)")
    args = ap.parse_args()

    robot = SimRobot() if args.sim else Robot(args.port, **CALIB)
    with robot:
        if not args.sim:
            print("raw ticks:", robot.read_raw())
        print("q now [deg]:", np.rad2deg(robot.read_q()))

        home = np.array([150.0, 0.0, 120.0])
        q = move_to_point(robot, home)
        print("home q [deg]:", np.rad2deg(q), "tip:", T04(q)[:3, 3])

        _, P = circle_points(100)
        current_speed = robot.get_speed()
        print("Motor speed 1", current_speed)
        robot.set_speed(1000)
        
        current_speed = robot.get_speed()
        print("Motor speed 2", current_speed)
        for qi in ik_path(P):
            robot.move_q(qi, wait = True, timeout=1/400)
        print("circle done")
        
        current_speed = robot.get_speed()
        print("Motor speed 3", current_speed)

        if args.cam is not None:
            run_camera(robot, args.cam)


def run_camera(robot, index):
    import cv2
    from robotics.vision import open_camera, pixel_to_plane

    cap = open_camera(index)
    clicked = {}
    cv2.namedWindow("cam")
    cv2.setMouseCallback("cam", lambda e, u, v, *_: clicked.update(uv=(u, v))
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if "uv" in clicked:
            u, v = clicked["uv"]
            p = pixel_to_plane(u, v, K, robot.read_q(), plane_z=TABLE_Z)
            cv2.circle(frame, (u, v), 5, (0, 0, 255), 2)
            txt = "no hit" if p is None else f"({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm"
            cv2.putText(frame, txt, (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("cam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
