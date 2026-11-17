"""Problems 1-10, ported from Robotics_Project.m.

Run:  python problems_1_10.py          (shows figures)
      python problems_1_10.py --save   (writes PNGs to ./figures instead)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

SAVE = "--save" in sys.argv
if SAVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robotics import (D_DEFAULT as d, forward_kinematics, origins, inverse_kinematics,
                      jacobian_stylus, jacobian_camera, reduced_jacobian,
                      circle_points, ik_path, circle_trajectory, T04)
from robotics.dynamics import build_model

np.set_printoptions(precision=4, suppress=True, linewidth=120)
FIG_DIR = Path(__file__).parent / "figures"
PI_TICKS = ([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])


def finish(fig, name):
    if SAVE:
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- Problem 1
print("Problem 1: T04 and T05 at q = 0")
Ts = forward_kinematics(np.zeros(4), d)
print("T04 =\n", Ts[4]); print("T05 =\n", Ts[5])

# ---------------------------------------------------------------- Problem 2
q_test = np.array([0.3, -0.4, -0.9, -0.5])
T = T04(q_test, d)
q_back = inverse_kinematics(T[:3, 3], T[2, 0], d)
print("\nProblem 2: IK round trip  q =", q_test, " -> IK(FK(q)) =", q_back)

# ---------------------------------------------------------------- Problem 3
phis, p0 = circle_points(37)
q_circ = ik_path(p0, 0.0, d)                       # 37 x 4
O = np.array([origins(q, d) for q in q_circ])      # 37 x 3 x 6
print("\nProblem 3: max tip error on circle [mm]:",
      np.abs(O[:, :, 4].T - p0).max())

fig = plt.figure(32)
ax = fig.add_subplot(projection="3d")
ax.plot(*p0, "k--", lw=1)
for i in range(37):
    ax.plot(*O[i, :, :5], "-", color=(0, 0.45, 0.74, 0.25))
for i in (0, 9, 18, 27):
    ax.plot(*O[i, :, :5], "-o", lw=2)
ax.plot(*O[:, :, 4].T, "r.", ms=8)
ax.set(xlabel="x0 [mm]", ylabel="y0 [mm]", zlabel="z0 [mm]",
       title="Robot configurations along the circle")
ax.set_aspect("equal")
finish(fig, "p3_configurations")

# ---------------------------------------------------------------- Problem 4
idx5 = [0, 9, 18, 27, 36]
for k in idx5:
    print(f"\nProblem 4: phi = {phis[k]:.4f} rad")
    print("J04 =\n", jacobian_stylus(q_circ[k], d))
    print("J05 =\n", jacobian_camera(q_circ[k], d))

# ---------------------------------------------------------------- Problem 5
qdot = np.linalg.solve(reduced_jacobian(q_circ[9], d), [0, -3, 0, 0])
print("\nProblem 5: qdot at phi = pi/2 for v = [0,-3,0] mm/s:", qdot)

# ------------------------------------------------------------ Problem 6 + 7
traj, q_via, qd_via = circle_trajectory(d)
print("\nProblem 6: via-point joint velocities (rows = via points)\n", qd_via)
for s in range(4):
    print(f"segment {s + 1} coefficients a0..a5 (columns = joints)\n", traj.A[s])

tsim = np.arange(0, 8.0001, 0.1)
q_traj, qd_traj, qdd_traj = traj.sample(tsim)

fig, ax = plt.subplots(num=7)
ax.plot(tsim, q_traj.T); ax.legend(["q1", "q2", "q3", "q4"])
ax.set(xlabel="t [s]", ylabel="q [rad]", title="Interpolated joint trajectory"); ax.grid()
finish(fig, "p7_joints")

tip = np.array([T04(q_traj[:, i], d)[:3, 3] for i in range(len(tsim))]).T
err = np.linalg.norm(tip - (np.array([150, 0, 120])[:, None]), axis=0) - 32
print("\nProblem 7: radial deviation from circle [mm]: max", np.abs(err).max())
fig, ax = plt.subplots(num=71)
ax.plot(tip[1], tip[2], label="Trajectory"); ax.plot(p0[1], p0[2], label="Desired circle")
ax.set(xlabel="y0 [mm]", ylabel="z0 [mm]"); ax.set_aspect("equal"); ax.legend(); ax.grid()
finish(fig, "p7_path")

# ---------------------------------------------------------------- Problem 8
dm = d / 1000
cond_circ = [np.linalg.cond(reduced_jacobian(q, dm)) for q in q_circ]
Jt = [reduced_jacobian(q_traj[:, i], dm) for i in range(len(tsim))]
cond_traj = [np.linalg.cond(J) for J in Jt]
det_traj = [np.linalg.det(J) for J in Jt]
eigs = np.array([np.linalg.eigvalsh(J @ J.T) for J in Jt])
print(f"\nProblem 8: cond along circle {min(cond_circ):.1f}..{max(cond_circ):.1f}, "
      f"along trajectory {min(cond_traj):.1f}..{max(cond_traj):.1f}, "
      f"min eig(JJ^T) {eigs.min():.4f}")
phi_traj = np.linspace(0, 2 * np.pi, len(tsim))
fig, axs = plt.subplots(2, 2, num=8, figsize=(10, 7))
for ax, (x, y, t) in zip(axs.flat, [
        (phis, cond_circ, "Condition number, 37-point circle"),
        (phi_traj, cond_traj, "Condition number, continuous trajectory"),
        (phi_traj, det_traj, "Determinant"),
        (phi_traj, eigs.min(1), "Lowest eigenvalue of JJᵀ")]):
    ax.plot(x, y); ax.set_title(t); ax.set_xticks(*PI_TICKS); ax.set_xlabel("φ [rad]"); ax.grid()
fig.tight_layout()
finish(fig, "p8_singularities")

# ---------------------------------------------------------------- Problem 9
F = np.array([0, 0, -1, 0, 0, 0])
torques = np.array([jacobian_stylus(q, dm).T @ F for q in q_circ])
fig, ax = plt.subplots(num=91)
ax.plot(phis, torques); ax.legend(["τ1", "τ2", "τ3", "τ4"])
ax.set(xlabel="φ [rad]", ylabel="Torque [Nm]", title="Static torques for 1 N on end-effector")
ax.set_xticks(*PI_TICKS); ax.grid()
finish(fig, "p9_static_torques")

# --------------------------------------------------------------- Problem 10
print("\nProblem 10: building symbolic dynamics model ...")
model = build_model(tuple(dm))
Dq = model["D"](q_circ[0])
print("D(q) symmetric:", np.allclose(Dq, Dq.T), " positive definite:",
      bool(np.all(np.linalg.eigvalsh(Dq) > 0)))
tau = np.array([model["tau"](q_traj[:, i], qd_traj[:, i], qdd_traj[:, i])
                for i in range(len(tsim))])
print("qdd at t=0, 3.9, 8:", qdd_traj[:, 0], qdd_traj[:, 39], qdd_traj[:, 80])
fig, ax = plt.subplots(num=11)
ax.plot(phi_traj, tau); ax.legend(["τ1", "τ2", "τ3", "τ4"])
ax.set(xlabel="φ [rad]", ylabel="Torque [Nm]", title="Joint torques along trajectory (Euler-Lagrange)")
ax.set_xticks(*PI_TICKS); ax.grid()
finish(fig, "p10_dynamic_torques")

if not SAVE:
    plt.show()
