import dataclasses
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import analyze_vsd as av
from common import ARM_SIDES, OUTPUT_DIR, SCENE_XML, arm_dofs, site_name
from run_vsd import GAINS, Logger, Ry
from vsd_controller import Target, VSDController, VSDGains

OUT = os.path.join(OUTPUT_DIR, "stages")
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
SIDE = "left"

OFF = dataclasses.replace(GAINS, k=0.0, alpha=0.0, zeta=0.0, use_orientation=False,
                          use_gravity=False, use_friction_comp=False, k_rep=0.0)


def _quat2mat(qt):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, qt)
    return R.reshape(3, 3)


def _rot(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _hom(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def fk_chain(m, side, q):
    site = m.site(site_name(side)).id
    chain = []
    b = m.site_bodyid[site]
    while b != 0:
        chain.append(b)
        b = m.body_parentid[b]
    chain.reverse()
    T = np.eye(4)
    origins, axes = [], []
    k = 0
    for b in chain:
        T = T @ _hom(_quat2mat(m.body_quat[b]), m.body_pos[b])
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            jp, ax = m.jnt_pos[j], m.jnt_axis[j]
            origins.append((T @ np.append(jp, 1))[:3])
            axes.append(T[:3, :3] @ ax)
            T = T @ _hom(np.eye(3), jp) @ _hom(_rot(ax, q[k]), np.zeros(3)) @ _hom(np.eye(3), -jp)
            k += 1
    T = T @ _hom(_quat2mat(m.site_quat[site]), m.site_pos[site])
    return T, np.array(origins), np.array(axes)


def verify_kinematics(m, n=200, seed=1):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    res = {}
    for side in ARM_SIDES:
        dof = arm_dofs(m, side)
        qadr = m.jnt_qposadr[m.dof_jntid[dof]]
        sid = m.site(site_name(side)).id
        ep = er = ejv = ejw = 0.0
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(n):
            q = rng.uniform(-np.pi, np.pi, 7)
            d.qpos[:] = m.qpos0
            d.qpos[qadr] = q
            mujoco.mj_forward(m, d)
            T, o, z = fk_chain(m, side, q)
            pE = T[:3, 3]
            Jv = np.stack([np.cross(z[i], pE - o[i]) for i in range(7)], axis=1)
            Jw = z.T
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            ep = max(ep, np.abs(pE - d.site_xpos[sid]).max())
            er = max(er, np.abs(T[:3, :3] - d.site_xmat[sid].reshape(3, 3)).max())
            ejv = max(ejv, np.abs(Jv - jacp[:, dof]).max())
            ejw = max(ejw, np.abs(Jw - jacr[:, dof]).max())
        res[side] = {"fk_pos_max_err_m": ep, "fk_rot_max_err": er,
                     "jv_max_err": ejv, "jw_max_err": ejw, "n_samples": n}
    return res


def potential(m, d):
    mujoco.mj_kinematics(m, d)
    return float(np.sum(m.body_mass * (-m.opt.gravity[2]) * d.xipos[:, 2]))


def verify_gravity(m, n=50, h=1e-6, seed=2):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    g = np.zeros(m.nv)
    worst, gmax = 0.0, 0.0
    for _ in range(n):
        d.qpos[:] = m.qpos0
        for side in ARM_SIDES:
            dof = arm_dofs(m, side)
            d.qpos[m.jnt_qposadr[m.dof_jntid[dof]]] = rng.uniform(-np.pi, np.pi, 7)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        mujoco.mj_rne(m, d, 0, g)
        for side in ARM_SIDES:
            for i in arm_dofs(m, side):
                a = m.jnt_qposadr[m.dof_jntid[i]]
                q0 = d.qpos[a]
                d.qpos[a] = q0 + h
                up = potential(m, d)
                d.qpos[a] = q0 - h
                um = potential(m, d)
                d.qpos[a] = q0
                worst = max(worst, abs((up - um) / (2 * h) - g[i]))
                gmax = max(gmax, abs(g[i]))
    return {"max_abs_err_Nm": worst, "max_abs_g_Nm": gmax, "n_samples": n}


def run_stage(name, gains, duration, reach_after=None, use_ori=False, push=None):
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    ctrl = VSDController(m, gains)
    ctrl.hold_fingers(d.qpos)
    home = {s: Target(d.site_xpos[a.site].copy(), d.site_xmat[a.site].reshape(3, 3).copy())
            for s, a in ctrl.arms.items()}
    reach = {s: Target(np.array([0.55, 0.17 if s == "left" else -0.17, 0.95]),
                       (Ry(np.deg2rad(30)) @ home[s].R) if use_ori else None) for s in ARM_SIDES}
    hold = {s: Target(home[s].pos, home[s].R if use_ori else None) for s in ARM_SIDES}
    bracelet = m.body(f"bracelet_link_{SIDE[0]}").id
    log = Logger()
    null_speed, p0 = [], d.site_xpos[ctrl.arms[SIDE].site].copy()
    a = ctrl.arms[SIDE]
    for _ in range(int(round(duration / m.opt.timestep))):
        tgt = reach if (reach_after is not None and d.time >= reach_after - 1e-9) else hold
        ctrl.set_targets(tgt)
        d.ctrl[:] = ctrl.compute(d)
        d.xfrc_applied[:] = 0
        if push is not None and push[0] <= d.time < push[1]:
            d.xfrc_applied[bracelet, :3] = push[2]
        log.add(d.time, "reach" if tgt is reach else "hold", ctrl.info, d.ncon)
        _, _, J, Jw = ctrl.ee_state(d, a)
        Jf = J if not use_ori else np.vstack([J, Jw])
        qd = d.qvel[a.dof]
        null_speed.append(np.linalg.norm(qd - np.linalg.pinv(Jf) @ (Jf @ qd)))
        mujoco.mj_step(m, d)
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    log.rows["null_speed"] = null_speed
    log.rows["p0"] = [p0] * len(null_speed)
    log.save(os.path.join(out, "log.npz"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"name": name, "gains": dataclasses.asdict(gains), "duration": duration,
                   "reach_after": reach_after, "use_ori": use_ori,
                   "push": None if push is None else [push[0], push[1], list(push[2])]}, f, indent=2)
    return dict(np.load(os.path.join(out, "log.npz")))


STAGES = {
    "w3_zero": dict(gains=dataclasses.replace(OFF, actuator_mode="torque"), duration=1.0),
    "w3_g_current": dict(gains=dataclasses.replace(OFF, use_gravity=True, actuator_mode="current"),
                         duration=3.0, push=(1.0, 1.1, (10.0, 0, 0))),
    "w3_gf_current": dict(gains=dataclasses.replace(OFF, use_gravity=True, use_friction_comp=True,
                                                    actuator_mode="current"),
                          duration=3.0, push=(1.0, 1.1, (10.0, 0, 0))),
    "w3_g_torque": dict(gains=dataclasses.replace(OFF, use_gravity=True, actuator_mode="torque"),
                        duration=3.0, push=(1.0, 1.1, (10.0, 0, 0))),
    "w5_spring": dict(gains=dataclasses.replace(OFF, use_gravity=True, k=GAINS.k, f_max=GAINS.f_max),
                      duration=4.0, reach_after=0.5),
    "w5_spring_c0": dict(gains=dataclasses.replace(OFF, use_gravity=True, k=GAINS.k, f_max=GAINS.f_max,
                                                   alpha=GAINS.alpha),
                         duration=4.0, reach_after=0.5),
    "w5_spring_c0_task": dict(gains=dataclasses.replace(OFF, use_gravity=True, k=GAINS.k, f_max=GAINS.f_max,
                                                        alpha=GAINS.alpha, zeta=GAINS.zeta),
                              duration=4.0, reach_after=0.5),
    "w5_full": dict(gains=dataclasses.replace(GAINS, k_rep=0.0), duration=4.0, reach_after=0.5, use_ori=True),
}


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


def _legend(ax, labels, loc="best"):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=SLOT[i], lw=2, label=l) for i, l in enumerate(labels)],
              loc=loc, fontsize=9)


def figures(L):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (run, lbl) in enumerate([("w3_zero", "Zero torque u = 0"), ("w3_g_torque", "Gravity comp. u = g(q)")]):
        t, p = L[run]["t"], L[run][f"{SIDE}_p"]
        sel = t < 1.0
        ax.plot(t[sel], (p[sel, 2] - p[0, 2]) * 1e3, color=SLOT[i])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("End-effector height change Δz [mm]")
    ax.set_title("Zero torque and gravity compensation (left arm, no external force)")
    _legend(ax, ["Zero torque u = 0", "Gravity comp. u = g(q)"], loc="lower left")
    _save(fig, "w3_gravity.png")

    runs = [("w3_g_current", "Gravity comp. (current mode, friction present)"),
            ("w3_gf_current", "Gravity + friction comp. (current mode)"),
            ("w3_g_torque", "Gravity comp. (torque mode, torque-sensor loop)")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (run, _) in enumerate(runs):
        t, p = L[run]["t"], L[run][f"{SIDE}_p"]
        axs[0].plot(t, np.linalg.norm(p - p[0], axis=1) * 1e3, color=SLOT[i])
        axs[1].plot(p[:, 0], p[:, 1], color=SLOT[i], lw=1.8)
    axs[0].axvspan(1.0, 1.1, color=av.GRID, zorder=0)
    axs[0].text(1.12, axs[0].get_ylim()[1] * 0.97, "push 10 N, 0.1 s", ha="left", va="top", fontsize=8, color=av.INK2)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel(r"End-effector displacement $|p - p_0|$ [mm]")
    axs[0].set_title("End-effector displacement after a push (left arm)")
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_aspect("equal", adjustable="datalim")
    axs[1].set_title("End-effector path (top view)")
    _legend(axs[0], [r for _, r in runs], loc="center left")
    fig.tight_layout()
    _save(fig, "w3_push.png")

    runs = [("w5_spring", r"Spring only $J^\top k\Delta p$"),
            ("w5_spring_c0", r"+ joint damping $C_0\dot q$"),
            ("w5_spring_c0_task", r"+ task-space damping $\zeta\sqrt{k}\dot p$")]
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    a1.sharex(a0)
    for i, (run, _) in enumerate(runs):
        t = L[run]["t"]
        ep = np.linalg.norm(L[run][f"{SIDE}_p"] - L[run][f"{SIDE}_pd"], axis=1) * 1e3
        a0.semilogy(t, np.maximum(ep, 1e-3), color=SLOT[i])
        a1.plot(t, L[run]["null_speed"], color=SLOT[i])
        a2.plot(L[run][f"{SIDE}_p"][:, 0], L[run][f"{SIDE}_p"][:, 1], color=SLOT[i], lw=1.6)
    p0, pd = L[runs[0][0]][f"{SIDE}_p"][0], L[runs[0][0]][f"{SIDE}_pd"][-1]
    a2.plot([p0[0], pd[0]], [p0[1], pd[1]], color=av.MUTED, lw=1, ls=(0, (4, 3)))
    a2.plot(pd[0], pd[1], "x", color=av.INK, ms=8, mew=2)
    a0.set_ylabel("|Δp| [mm] (log)")
    a1.set_ylabel(r"Self-motion speed $\|(I-J^{+}J)\dot q\|$ [rad/s]")
    a1.set_xlabel("Time [s]")
    a0.set_title("Adding VSD terms one at a time (left arm)")
    a2.set_xlabel("x [m]")
    a2.set_ylabel("y [m]")
    a2.set_aspect("equal", adjustable="datalim")
    a2.set_title("End-effector path (top view)")
    for ax in (a0, a1):
        ax.axvline(0.5, color=av.MUTED, lw=1, ls=(0, (2, 2)), zorder=0)
    _legend(a0, [r for _, r in runs], loc="lower left")
    fig.tight_layout()
    _save(fig, "w5_stages.png")


def summarize(L):
    s = {}
    for run in ("w3_zero",):
        p = L[run][f"{SIDE}_p"]
        s[run] = {"z_drop_mm": float((p[0, 2] - p[-1, 2]) * 1e3), "duration_s": float(L[run]["t"][-1])}
    for run in ("w3_g_torque",):
        t, p = L[run]["t"], L[run][f"{SIDE}_p"]
        s[run + "_hold"] = {"max_drift_before_push_mm": float(np.abs(p[t < 1.0] - p[0]).max() * 1e3)}
    for run in ("w3_g_current", "w3_gf_current", "w3_g_torque"):
        t, p = L[run]["t"], L[run][f"{SIDE}_p"]
        disp = np.linalg.norm(p - p[0], axis=1)
        v = np.linalg.norm(L[run][f"{SIDE}_pdot"], axis=1)
        moving = np.where((t > 1.1) & (v > 1e-3))[0]
        s[run] = {"final_disp_mm": float(disp[-1] * 1e3), "max_disp_mm": float(disp.max() * 1e3),
                  "drift_before_push_mm": float(disp[t < 1.0].max() * 1e3),
                  "stop_time_s": float(t[moving[-1]] - 1.1) if len(moving) else 0.0}
    for run in ("w5_spring", "w5_spring_c0", "w5_spring_c0_task", "w5_full"):
        t = L[run]["t"]
        ep = np.linalg.norm(L[run][f"{SIDE}_p"] - L[run][f"{SIDE}_pd"], axis=1)
        tail = t >= t[-1] - 0.5
        p = L[run][f"{SIDE}_p"]
        i0 = np.searchsorted(t, 0.5)
        a, b = p[i0], L[run][f"{SIDE}_pd"][-1]
        ab = (b - a) / np.linalg.norm(b - a)
        dev = np.linalg.norm((p[i0:] - a) - np.outer((p[i0:] - a) @ ab, ab), axis=1)
        s[run] = {"err_tail_mean_mm": float(ep[tail].mean() * 1e3), "err_tail_max_mm": float(ep[tail].max() * 1e3),
                  "null_speed_tail": float(np.mean(L[run]["null_speed"][tail])),
                  "max_path_dev_mm": float(dev.max() * 1e3),
                  "max_contacts": int(L[run]["ncon"].max())}
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    ver = {"week2_kinematics": verify_kinematics(m), "week3_gravity": verify_gravity(m)}
    L = {}
    for name, cfg in STAGES.items():
        print("run", name)
        L[name] = run_stage(name, **cfg)
    ver["stages"] = summarize(L)
    with open(os.path.join(OUT, "verification.json"), "w") as f:
        json.dump(ver, f, indent=2, ensure_ascii=False)
    figures(L)
    print(json.dumps(ver, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
