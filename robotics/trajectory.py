"""Path and trajectory planning (Problems 3, 6, 7)."""
import numpy as np

from .kinematics import D_DEFAULT, inverse_kinematics, joint_velocities

CIRCLE_CENTER = np.array([150.0, 0.0, 120.0])  # [mm]
CIRCLE_RADIUS = 32.0                           # [mm]


def circle_points(n=37, center=CIRCLE_CENTER, radius=CIRCLE_RADIUS):
    """Problem 3: n points (phi = 0..2pi inclusive) on a circle in the y-z plane.

    Returns (phis, P) where P is 3 x n.
    """
    phis = np.linspace(0, 2 * np.pi, n)
    P = center[:, None] + radius * np.vstack([np.zeros(n), np.cos(phis), np.sin(phis)])
    return phis, P


def ik_path(P, x4z=0.0, d=D_DEFAULT):
    """Inverse kinematics for each column of P -> (n x 4) joint array."""
    return np.array([inverse_kinematics(P[:, i], x4z, d) for i in range(P.shape[1])])


def quintic_coeffs(t0, tf, q0, v0, a0, qf, vf, af):
    """Coefficients a0..a5 of a quintic polynomial meeting the boundary conditions."""
    M = np.array([
        [1, t0, t0**2, t0**3,    t0**4,     t0**5],
        [0, 1,  2*t0,  3*t0**2,  4*t0**3,   5*t0**4],
        [0, 0,  2,     6*t0,     12*t0**2,  20*t0**3],
        [1, tf, tf**2, tf**3,    tf**4,     tf**5],
        [0, 1,  2*tf,  3*tf**2,  4*tf**3,   5*tf**4],
        [0, 0,  2,     6*tf,     12*tf**2,  20*tf**3],
    ], dtype=float)
    return np.linalg.solve(M, [q0, v0, a0, qf, vf, af])


class QuinticTrajectory:
    """Piecewise quintic joint trajectory through via points (Problem 6).

    q_via  : (m x 4) joint positions at the via points
    qd_via : (m x 4) joint velocities at the via points
    times  : (m,) times of the via points
    Accelerations at via points are set to zero.
    """

    def __init__(self, q_via, qd_via, times):
        self.times = np.asarray(times, dtype=float)
        q_via, qd_via = np.asarray(q_via), np.asarray(qd_via)
        nseg, nj = len(times) - 1, q_via.shape[1]
        # A[seg] is 6 x nj (same layout as MATLAB A(:,:,seg))
        self.A = np.zeros((nseg, 6, nj))
        for s in range(nseg):
            for k in range(nj):
                self.A[s, :, k] = quintic_coeffs(
                    self.times[s], self.times[s + 1],
                    q_via[s, k], qd_via[s, k], 0.0,
                    q_via[s + 1, k], qd_via[s + 1, k], 0.0)

    def segment(self, t):
        s = int(np.searchsorted(self.times, t, side="right")) - 1
        return min(max(s, 0), len(self.A) - 1)

    def evaluate(self, t):
        """Return (q, qd, qdd) at time t."""
        A = self.A[self.segment(t)]
        tv = np.array([1, t, t**2, t**3, t**4, t**5])
        tvd = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        tvdd = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        return tv @ A, tvd @ A, tvdd @ A

    def sample(self, tsim):
        """Evaluate on an array of times -> q, qd, qdd each (4 x len(tsim))."""
        out = np.array([np.column_stack(self.evaluate(t)) for t in tsim])  # N x 4 x 3
        return out[:, :, 0].T, out[:, :, 1].T, out[:, :, 2].T


def circle_trajectory(d=D_DEFAULT, idx=(0, 9, 18, 27, 36), times=(0, 2, 4, 6, 8)):
    """Problem 6 set-up: 4 quintic segments through phi = 0, pi/2, pi, 3pi/2, 2pi."""
    phis, P = circle_points()
    q_circ = ik_path(P, 0.0, d)
    # Cartesian velocity constraints at the via points [mm/s]
    V = np.array([[0, 0, 0], [0, -27, 0], [0, 0, -27], [0, 27, 0], [0, 0, 0]], dtype=float)
    q_via = q_circ[list(idx)]
    qd_via = np.array([joint_velocities(q_via[i], V[i], 0.0, d) for i in range(len(idx))])
    return QuinticTrajectory(q_via, qd_via, times), q_via, qd_via
