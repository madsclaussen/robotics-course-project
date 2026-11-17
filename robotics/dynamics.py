"""Euler-Lagrange dynamics of the arm (Problem 10), built symbolically with sympy.

Everything here is in SI units (metres, kg, N, Nm).
The symbolic model is built once (takes a few seconds) and lambdified.
"""
import functools

import numpy as np
import sympy as sp

from .kinematics import D_DEFAULT

M_LINK = np.array([60, 80, 80, 40]) / 1000.0          # [kg]
I0 = 2e-5                                             # [kg m^2]
INERTIA_BODY = [                                      # in link frames
    np.diag([1.0, 0.4, 0.9]) * I0,
    np.diag([0.45, 1.4, 1.2]) * I0,
    np.diag([0.45, 1.4, 1.2]) * I0,
    np.diag([0.5, 0.5, 0.5]) * I0,
]
CM_LOCAL = [np.array(v) / 1000.0 for v in                 # CoM in link frames [m]
            ([0, -20, 0], [-30, 0, 0], [-30, 0, 0], [-25, 15, 0])]
GRAVITY = np.array([0.0, 0.0, -9.82])


def _sym_dh(theta, d, a, alpha):
    ct, st, ca, sa = sp.cos(theta), sp.sin(theta), sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([[ct, -st * ca, st * sa, a * ct],
                      [st, ct * ca, -ct * sa, a * st],
                      [0, sa, ca, d],
                      [0, 0, 0, 1]])


@functools.lru_cache(maxsize=None)
def build_model(d=tuple(D_DEFAULT / 1000.0)):
    """Return dict of lambdified functions D(q), C(q,qd), g(q), tau(q,qd,qdd)."""
    q = sp.symbols("q1:5", real=True)
    qd = sp.symbols("qd1:5", real=True)
    qdd = sp.symbols("qdd1:5", real=True)
    d1, d2, d3, d4 = [sp.nsimplify(x) for x in d]
    pi = sp.pi

    Hs = [_sym_dh(q[0], d1, 0, pi / 2), _sym_dh(q[1] + pi / 2, 0, d2, 0),
          _sym_dh(q[2], 0, d3, 0), _sym_dh(q[3], 0, d4, 0)]
    Ts = [sp.eye(4)]
    for H in Hs:
        Ts.append(Ts[-1] * H)
    z = [T[:3, 2] for T in Ts]
    o = [T[:3, 3] for T in Ts]
    R = [T[:3, :3] for T in Ts]

    n = 4
    Dq = sp.zeros(n, n)
    Pot = 0
    g_vec = sp.Matrix([sp.nsimplify(x) for x in GRAVITY])
    for i in range(n):
        cm = o[i + 1] + R[i + 1] * sp.Matrix([sp.nsimplify(x) for x in CM_LOCAL[i]])
        Jv = sp.zeros(3, n)
        Jw = sp.zeros(3, n)
        for j in range(i + 1):
            Jv[:, j] = z[j].cross(cm - o[j])
            Jw[:, j] = z[j]
        Ib = sp.Matrix(INERTIA_BODY[i]).applyfunc(sp.nsimplify)
        Iw = R[i + 1] * Ib * R[i + 1].T
        m = sp.nsimplify(M_LINK[i])
        Dq += m * Jv.T * Jv + Jw.T * Iw * Jw
        Pot += -m * (g_vec.T * cm)[0]

    # Christoffel symbols c_ijk and Coriolis matrix C_kj = sum_i c_ijk qd_i
    dD = [Dq.diff(qi) for qi in q]
    C = sp.zeros(n, n)
    for k in range(n):
        for j in range(n):
            C[k, j] = sum(sp.Rational(1, 2) * (dD[i][k, j] + dD[j][k, i] - dD[k][i, j]) * qd[i]
                          for i in range(n))
    g = sp.Matrix([Pot.diff(qk) for qk in q])

    fD = sp.lambdify([q], Dq, "numpy", cse=True)
    fC = sp.lambdify([q, qd], C, "numpy", cse=True)
    fg = sp.lambdify([q], g, "numpy", cse=True)

    def tau(qv, qdv, qddv):
        qv, qdv, qddv = map(np.asarray, (qv, qdv, qddv))
        return (np.asarray(fD(qv), float) @ qddv
                + np.asarray(fC(qv, qdv), float) @ qdv
                + np.asarray(fg(qv), float).ravel())

    return {"D": lambda qv: np.asarray(fD(qv), float),
            "C": lambda qv, qdv: np.asarray(fC(qv, qdv), float),
            "g": lambda qv: np.asarray(fg(qv), float).ravel(),
            "tau": tau,
            "symbolic": {"D": Dq, "C": C, "g": g, "P": Pot, "q": q, "qd": qd}}
