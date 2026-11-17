# Python port of Robotics_Project.m

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python problems_1_10.py          # problems 1-10, shows figures
.venv/bin/python problems_1_10.py --save   # writes figures/*.png
.venv/bin/python problem11_start.py --sim  # problem 11 scaffold without hardware
```

| module | content | MATLAB counterpart |
|---|---|---|
| `robotics/kinematics.py` | DH, FK (T04 stylus, T05 camera), IK, J4/J5, reduced 4x4 Jacobian | Problems 1, 2, 4, 5 |
| `robotics/trajectory.py` | circle points, quintic via-point trajectory | Problems 3, 6, 7 |
| `robotics/dynamics.py` | sympy Euler-Lagrange model: D(q), C(q,qd), g(q), tau | Problem 10 |
| `robotics/hardware.py` | Dynamixel `Robot` (Annex C) + `SimRobot` stand-in | Problem 11 |
| `robotics/vision.py` | OpenCV camera, camera pose, pixel -> table plane, multi-view triangulation | Problem 11 |

Units: kinematics in mm (pass `d/1000` for metres, as Problems 8-9 do); dynamics in SI.

Before using the real robot: calibrate `CALIB` (signs/offsets/limits) in
`problem11_start.py`, the camera intrinsics `K` and the camera orientation
`R_5C_DEFAULT` in `robotics/vision.py`.
