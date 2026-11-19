import sys
from pathlib import Path
import time

import numpy as np
import matplotlib
import cv2

SAVE = "--save" in sys.argv
if SAVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

from robotics import (D_DEFAULT as d, forward_kinematics, origins, inverse_kinematics,
                      jacobian_stylus, jacobian_camera, reduced_jacobian,
                      circle_points, ik_path, circle_trajectory, is_reachable, T04)
from robotics.dynamics import build_model
from robotics.hardware import Robot, SimRobot
import cv2
from robotics.vision import open_camera, grab, pixel_to_plane, find_edges, uv_coordinates

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
    print("main")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--cam", type=int, default=None, help="camera index (omit to skip)")
    args = ap.parse_args()

    robot = SimRobot() if args.sim else Robot(args.port, **CALIB)
    with robot:
        if not args.sim:
            print("raw ticks:", robot.read_raw())

        # print("q now [deg]:", np.rad2deg(robot.read_q()))
        # home_q = np.array([0,0,0,0])
        # home_p = forward_kinematics(home_q)
        
        # robot.move_q(home_q)
        # time.sleep(1)

        circle_center = np.array([150.0, 0.0, 120.0])

        q = move_to_point(robot, circle_center, x4z= -0.8)

        print("home q [deg]:", np.rad2deg(q), "tip:", T04(q)[:3, 3])

        if args.cam is None:
            raise("Camera not specified")
        cap = open_camera(args.cam)

        cv2.namedWindow("cam")

        while True:
            frame = grab(cap)
            edges = find_edges(frame)
            uv = uv_coordinates(edges)
            

            cv2.imshow("cam", find_edges(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        _, P = circle_points(37)
        for qi in ik_path(P):
            robot.move_q(qi)
        print("circle done")






if __name__ == "__main__":
    main()
