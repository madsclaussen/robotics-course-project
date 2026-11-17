"""Forward/inverse/velocity kinematics of the 4-DOF course robot (Problems 1-5).

Port of Robotics_Project.m. All lengths are in millimetres unless a different
link-length vector `d` is passed (e.g. d / 1000 for metres).

Joint convention (Figure 1): q = 0 is the arm pointing straight up.
"""
import numpy as np

# Link lengths from Figure 1 [mm]: d1 (base height), d2, d3, d4 (stylus)
D_DEFAULT = np.array([50.0, 93.0, 93.0, 50.0])
# Camera (frame 5) offset expressed in frame 4 [mm]
CAM_OFFSET_DEFAULT = np.array([-15.0, 45.0, 0.0])


def dh_transform(theta, d, a, alpha):
    """Standard Denavit-Hartenberg homogeneous transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def link_transforms(q, d=D_DEFAULT, cam_offset=CAM_OFFSET_DEFAULT):
    """Return the individual transforms [H01, H12, H23, H34, H45]."""
    q1, q2, q3, q4 = q
    d1, d2, d3, d4 = d
    H45 = np.eye(4)
    H45[:3, 3] = cam_offset
    return [
        dh_transform(q1, d1, 0.0, np.pi / 2),
        dh_transform(q2 + np.pi / 2, 0.0, d2, 0.0),
        dh_transform(q3, 0.0, d3, 0.0),
        dh_transform(q4, 0.0, d4, 0.0),
        H45,
    ]


def forward_kinematics(q, d=D_DEFAULT, cam_offset=CAM_OFFSET_DEFAULT):
    """Cumulative transforms [T00, T01, T02, T03, T04, T05] (T00 = identity).

    T04 is the stylus tip, T05 the camera.
    """
    Ts = [np.eye(4)]
    for H in link_transforms(q, d, cam_offset):
        Ts.append(Ts[-1] @ H)
    return Ts


def T04(q, d=D_DEFAULT):
    return forward_kinematics(q, d)[4]


def T05(q, d=D_DEFAULT, cam_offset=CAM_OFFSET_DEFAULT):
    return forward_kinematics(q, d, cam_offset)[5]


def origins(q, d=D_DEFAULT, cam_offset=CAM_OFFSET_DEFAULT):
    """3x6 array of frame origins o0..o5 (columns), handy for plotting the arm."""
    return np.column_stack([T[:3, 3] for T in forward_kinematics(q, d, cam_offset)])


def inverse_kinematics(p, x4z=0.0, d=D_DEFAULT, elbow_up=True):
    """Problem 2: q = f(x4^0, o4^0).

    p    : stylus tip position o4^0 = [x, y, z]
    x4z  : z-component of the stylus direction x4^0 (in [-1, 1]);
           0 = stylus horizontal, -1 = pointing straight down.
    """
    D01, D12, D23, D34 = d
    p = np.asarray(p, dtype=float)

    q1 = np.arctan2(p[1], p[0])

    h = np.sqrt(max(0.0, 1.0 - x4z ** 2))
    x4 = np.array([np.cos(q1) * h, np.sin(q1) * h, x4z])

    o01 = np.array([0.0, 0.0, D01])
    o03 = p - D34 * x4
    o13 = o03 - o01

    a, b, c = D23, D12, np.linalg.norm(o13)

    gamma = np.arccos(np.clip((c**2 - a**2 - b**2) / (-2 * a * b), -1, 1))
    q3 = -(np.pi - gamma) if elbow_up else (np.pi - gamma)

    alpha = np.arccos(np.clip((a**2 - b**2 - c**2) / (-2 * b * c), -1, 1))
    phi = np.arctan2(o13[2], np.cos(q1) * o13[0] + np.sin(q1) * o13[1])
    q2 = -(np.pi / 2 - alpha - phi) if elbow_up else -(np.pi / 2 + alpha - phi)

    psi = np.arcsin(np.clip(x4z, -1, 1))
    q4 = psi - np.pi / 2 - q2 - q3
    return np.array([q1, q2, q3, q4])


def is_reachable(p, x4z=0.0, d=D_DEFAULT):
    """True if the wrist point is within reach of links 2+3."""
    D01, D12, D23, D34 = d
    p = np.asarray(p, dtype=float)
    q1 = np.arctan2(p[1], p[0])
    h = np.sqrt(max(0.0, 1.0 - x4z ** 2))
    x4 = np.array([np.cos(q1) * h, np.sin(q1) * h, x4z])
    c = np.linalg.norm(p - D34 * x4 - np.array([0, 0, D01]))
    return abs(D12 - D23) <= c <= D12 + D23


def _geometric_jacobian(Ts, n_idx):
    on = Ts[n_idx][:3, 3]
    J = np.zeros((6, 4))
    for i in range(4):
        z = Ts[i][:3, 2]
        o = Ts[i][:3, 3]
        J[:3, i] = np.cross(z, on - o)
        J[3:, i] = z
    return J


def jacobian_stylus(q, d=D_DEFAULT):
    """Problem 4: 6x4 geometric Jacobian J4 for the stylus tip (frame 4)."""
    return _geometric_jacobian(forward_kinematics(q, d), 4)


def jacobian_camera(q, d=D_DEFAULT, cam_offset=CAM_OFFSET_DEFAULT):
    """Problem 4: 6x4 geometric Jacobian J5 for the camera (frame 5)."""
    return _geometric_jacobian(forward_kinematics(q, d, cam_offset), 5)


def reduced_jacobian(q, d=D_DEFAULT):
    """Problem 5: square 4x4 Jacobian mapping qdot -> [v4; d(x4z)/dt].

    Last row uses Annex A: x4z_dot = [x4y, -x4x, 0] . omega.
    """
    T = T04(q, d)
    J = jacobian_stylus(q, d)
    row = np.array([T[1, 0], -T[0, 0], 0.0]) @ J[3:, :]
    return np.vstack([J[:3, :], row])


def joint_velocities(q, v, x4z_dot=0.0, d=D_DEFAULT):
    """Solve for qdot given tip velocity v (3,) and stylus-tilt rate."""
    return np.linalg.solve(reduced_jacobian(q, d), np.append(v, x4z_dot))
