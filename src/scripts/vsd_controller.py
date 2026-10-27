from dataclasses import dataclass, field

import mujoco
import numpy as np

from common import ARM_SIDES, arm_actuators, arm_dofs, site_name
from kinova_actuator import KinovaTorqueActuators


@dataclass
class VSDGains:
    k: float = 300.0
    zeta: float = 1.0
    alpha: float = 0.8
    damping_mode: str = "inertia"
    use_orientation: bool = True
    use_gravity: bool = True
    k_o: float = 10.0
    zeta_o: float = 0.5
    use_friction_comp: bool = True
    fric_eps: float = 0.02
    f_max: float = None
    m_max: float = None
    k_rep: float = 2.0
    r_th: float = 0.25
    actuator_mode: str = "torque"
    finger_kp: float = 2.0
    finger_kd: float = 0.05


@dataclass
class Target:
    pos: np.ndarray
    R: np.ndarray = field(default=None)


def _saturate(f, f_max):
    if f_max is None:
        return f
    n = np.linalg.norm(f)
    return f if n <= f_max else f * (f_max / n)


def orientation_error(R, Rd):
    return 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))


def inertia_bound(m, dof, n_samples=300, seed=0):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[m.dof_jntid[i]] for i in dof])
    lim = [m.jnt_limited[m.dof_jntid[i]] for i in dof]
    lo = np.array([m.jnt_range[m.dof_jntid[i]][0] if l else -np.pi for i, l in zip(dof, lim)])
    hi = np.array([m.jnt_range[m.dof_jntid[i]][1] if l else np.pi for i, l in zip(dof, lim)])
    H = np.zeros((m.nv, m.nv))
    HM = np.zeros((len(dof), len(dof)))
    for _ in range(n_samples):
        d.qpos[qadr] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, H, d.qM)
        HM = np.maximum(HM, np.abs(H[np.ix_(dof, dof)]))
    return HM


class Arm:
    def __init__(self, m, side, gains):
        self.side = side
        self.site = m.site(site_name(side)).id
        self.dof = arm_dofs(m, side)
        self.act = arm_actuators(m, side)
        self.gear = m.actuator_gear[self.act, 0].copy()
        self.ctrlrange = m.actuator_ctrlrange[self.act].copy()
        self.tau_max = self.ctrlrange[:, 1] * self.gear
        self.cv = m.dof_damping[self.dof].copy()
        self.cc = m.dof_frictionloss[self.dof].copy()
        self.HM = inertia_bound(m, self.dof)
        self.drive = KinovaTorqueActuators(m, self.dof, self.act, mode=gains.actuator_mode)
        c = gains.alpha * np.sqrt(self.HM.sum(axis=1))
        if gains.damping_mode == "uniform":
            c = np.full_like(c, c.mean())
        self.C0 = np.diag(c)


class VSDController:
    def __init__(self, m, gains=None):
        self.m = m
        self.gains = gains or VSDGains()
        self.arms = {s: Arm(m, s, self.gains) for s in ARM_SIDES}
        self.targets = {}
        self.g = np.zeros(m.nv)
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

        arm_acts = set(np.concatenate([a.act for a in self.arms.values()]).tolist())
        self.f_act = np.array([i for i in range(m.nu) if i not in arm_acts])
        jnt = m.actuator_trnid[self.f_act, 0]
        self.f_dof = m.jnt_dofadr[jnt]
        self.f_qadr = m.jnt_qposadr[jnt]
        self.f_qd = None
        self.info = {}

    def set_targets(self, targets):
        self.targets = targets

    def hold_fingers(self, qpos):
        self.f_qd = qpos[self.f_qadr].copy()

    def _gravity(self, d):
        qvel = d.qvel.copy()
        d.qvel[:] = 0
        mujoco.mj_rne(self.m, d, 0, self.g)
        d.qvel[:] = qvel

    def ee_state(self, d, arm):
        mujoco.mj_jacSite(self.m, d, self._jacp, self._jacr, arm.site)
        return (d.site_xpos[arm.site].copy(), d.site_xmat[arm.site].reshape(3, 3).copy(),
                self._jacp[:, arm.dof].copy(), self._jacr[:, arm.dof].copy())

    def compute(self, d):
        gn = self.gains
        self._gravity(d)
        ctrl = np.zeros(self.m.nu)
        states = {s: self.ee_state(d, a) for s, a in self.arms.items()}

        diff = states["right"][0] - states["left"][0]
        dist = np.linalg.norm(diff)
        f_rep = gn.k_rep * diff / dist**2 if dist < gn.r_th else np.zeros(3)
        self.info = {"dist": dist, "f_rep": f_rep}

        for s, arm in self.arms.items():
            p, R, J, Jw = states[s]
            tgt = self.targets[s]
            qd = d.qvel[arm.dof].copy()
            dp = p - tgt.pos
            pdot = J @ qd
            w = Jw @ qd

            tau_damp = -arm.C0 @ qd
            tau_spring = -J.T @ _saturate(gn.k * dp, gn.f_max)
            tau_tdamp = -J.T @ (gn.zeta * np.sqrt(gn.k) * pdot)
            e_o = orientation_error(R, tgt.R) if tgt.R is not None else np.zeros(3)
            tau_ori = (Jw.T @ (_saturate(gn.k_o * e_o, gn.m_max) - gn.zeta_o * np.sqrt(gn.k_o) * w)
                       if gn.use_orientation and tgt.R is not None else np.zeros(7))
            tau_g = self.g[arm.dof].copy() if gn.use_gravity else np.zeros(7)
            tau_fric = (arm.cc * np.tanh(qd / gn.fric_eps) + arm.cv * qd
                        if gn.use_friction_comp else np.zeros(7))
            tau_rep = J.T @ (f_rep if s == "right" else -f_rep)

            tau = tau_damp + tau_spring + tau_tdamp + tau_ori + tau_g + tau_fric + tau_rep
            u, tau_m, tau_s = arm.drive.command(d, tau)
            ctrl[arm.act] = u
            self.info[s] = dict(p=p, pd=tgt.pos.copy(), pdot=pdot, e_o=e_o, w=w,
                                q=d.qpos[[self.m.jnt_qposadr[self.m.dof_jntid[i]] for i in arm.dof]].copy(),
                                qd=qd, tau=tau, tau_m=tau_m, tau_s=tau_s,
                                tau_damp=tau_damp, tau_spring=tau_spring, tau_tdamp=tau_tdamp,
                                tau_ori=tau_ori, tau_g=tau_g, tau_fric=tau_fric, tau_rep=tau_rep)

        if self.f_qd is None:
            self.hold_fingers(d.qpos)
        q, qd = d.qpos[self.f_qadr], d.qvel[self.f_dof]
        ctrl[self.f_act] = (self.g[self.f_dof] + self.gains.finger_kp * (self.f_qd - q)
                            - self.gains.finger_kd * qd)
        return ctrl
