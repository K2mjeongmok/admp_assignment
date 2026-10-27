import mujoco
import numpy as np


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp_so3(xi):
    n = np.linalg.norm(xi)
    X = skew(xi)
    if n < 1e-9:
        return np.eye(3) + X, np.eye(3) + 0.5 * X
    alpha = np.sin(n) / n
    beta = 2.0 * (1.0 - np.cos(n)) / n**2
    R = np.eye(3) + alpha * X + 0.5 * beta * X @ X
    dexp = np.eye(3) + 0.5 * beta * X + (1.0 - alpha) / n**2 * X @ X
    return R, dexp


def exp_se3(lam):
    eta, xi = lam[:3], lam[3:]
    R, dexp = exp_so3(xi)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, dexp @ eta
    return T


def adjoint(T):
    R, r = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = skew(r) @ R
    A[3:, 3:] = R
    return A


class PoEPoint:
    def __init__(self, m, body, offset=(0.0, 0.0, 0.0)):
        self.m = m
        self.body = m.body(body).id
        self.offset = np.asarray(offset, float)
        chain, b = [], self.body
        while b != 0:
            chain.append(b)
            b = m.body_parentid[b]
        self.joints = [j for b in reversed(chain)
                       for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])]
        self.dofs = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joints])
        d0 = mujoco.MjData(m)
        d0.qpos[:] = 0
        mujoco.mj_kinematics(m, d0)
        self.S = np.array([np.concatenate([-np.cross(d0.xaxis[j], d0.xanchor[j]), d0.xaxis[j]])
                           for j in self.joints])
        R0 = d0.xmat[self.body].reshape(3, 3)
        self.M = np.eye(4)
        self.M[:3, 3] = d0.xpos[self.body] + R0 @ self.offset

    def fk_jacobian(self, qpos):
        T = np.eye(4)
        Js = np.zeros((6, len(self.joints)))
        for k, (S, a) in enumerate(zip(self.S, self.qadr)):
            Js[:, k] = adjoint(T) @ S
            T = T @ exp_se3(S * qpos[a])
        p = (T @ self.M)[:3, 3]
        Jp = Js[:3] - skew(p) @ Js[3:]
        return p, Jp


def verify(m, points, n=200, seed=3):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    jacp = np.zeros((3, m.nv))
    ep = ej = 0.0
    for _ in range(n):
        d.qpos[:] = rng.uniform(-np.pi, np.pi, m.nq)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        for P in points:
            p, Jp = P.fk_jacobian(d.qpos)
            R = d.xmat[P.body].reshape(3, 3)
            p_mj = d.xpos[P.body] + R @ P.offset
            mujoco.mj_jac(m, d, jacp, None, p_mj, P.body)
            full = np.zeros((3, m.nv))
            full[:, P.dofs] = Jp
            ep = max(ep, np.abs(p - p_mj).max())
            ej = max(ej, np.abs(full - jacp).max())
    return {"pos_max_err_m": ep, "jac_max_err": ej, "n_samples": n, "n_points": len(points)}
