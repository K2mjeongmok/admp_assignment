import dataclasses
import json
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares

import analyze_vsd as av
from common import ARM_SIDES, OUTPUT_DIR, SCENE_XML, arm_dofs
from kinova_actuator import KinovaTorqueActuators
from lie_jacobian import PoEPoint
from run_vsd import GAINS, Ry
from vsd_controller import Target, VSDController, _saturate, inertia_bound

OUT = os.path.join(OUTPUT_DIR, "contact")
FINGERS = ("ff", "mf", "rf", "th")
TIP_OFFSET = {"ff": 0.019, "mf": 0.019, "rf": 0.019, "th": 0.035}
DESK_TOP = 0.745
APPROACH_Z, TILT_DEG, ROLL_DEG, DZ = 0.91, 20.0, 25.0, -0.10
T_APPROACH, T_PRESS = 3.0, 3.0
SERVO_KP = np.array([2000.0] * 4 + [500.0] * 3)
SERVO_KD = np.array([100.0] * 4 + [50.0] * 3)
FINGER_TAU_MAX = 0.7
METHODS = {"servo": "Position tracking (joint servo)", "ee_vsd": "End-effector VSD", "tip_vsd": "Fingertip VSD"}
SLOT = {"servo": "#2a78d6", "ee_vsd": "#eb6834", "tip_vsd": "#1baf7a"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
R_PALM_DOWN = {"left": np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float),
               "right": np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float)}


def hand_joint_ids(m, side):
    return [j for j in range(m.njnt) if m.joint(j).name.startswith(side + "_") and "_joint_" not in m.joint(j).name]


def gravity(m, d, out):
    qvel = d.qvel.copy()
    d.qvel[:] = 0
    mujoco.mj_rne(m, d, 0, out)
    d.qvel[:] = qvel


class FingerHold:
    def __init__(self, m, qpos, kp=2.0, kd=0.05):
        self.acts = np.array([a for a in range(m.nu) if "_joint_" not in m.joint(m.actuator_trnid[a, 0]).name])
        jnt = m.actuator_trnid[self.acts, 0]
        self.dof, self.qadr = m.jnt_dofadr[jnt], m.jnt_qposadr[jnt]
        self.qd, self.kp, self.kd = qpos[self.qadr].copy(), kp, kd

    def ctrl(self, d, g):
        return g[self.dof] + self.kp * (self.qd - d.qpos[self.qadr]) - self.kd * d.qvel[self.dof]


class JointServo:
    def __init__(self, m, d, targets, drives):
        self.m, self.drives = m, drives
        self.hold = FingerHold(m, d.qpos)
        self.g = np.zeros(m.nv)
        self.qd_des = {s: self._ik(d, s, targets[s]) for s in ARM_SIDES}

    def _ik(self, d, side, tgt):
        m = self.m
        dd = mujoco.MjData(m)
        dd.qpos[:] = d.qpos
        dof = arm_dofs(m, side)
        qadr = m.jnt_qposadr[m.dof_jntid[dof]]
        sid = m.site(f"ee_site_{side[0]}").id

        def res(q):
            dd.qpos[qadr] = q
            mujoco.mj_kinematics(m, dd)
            return np.concatenate([10 * (dd.site_xpos[sid] - tgt.pos),
                                   (dd.site_xmat[sid].reshape(3, 3) - tgt.R).ravel()])
        return least_squares(res, d.qpos[qadr].copy()).x

    def compute(self, d):
        m = self.m
        gravity(m, d, self.g)
        ctrl = np.zeros(m.nu)
        for s in ARM_SIDES:
            dr = self.drives[s]
            q = d.qpos[m.jnt_qposadr[m.dof_jntid[dr.dofs]]]
            tau = SERVO_KP * (self.qd_des[s] - q) - SERVO_KD * d.qvel[dr.dofs] + self.g[dr.dofs]
            ctrl[dr.acts], _, _ = dr.command(d, tau)
        ctrl[self.hold.acts] = self.hold.ctrl(d, self.g)
        return ctrl


class EEVSD:
    def __init__(self, m, d, targets, drives):
        self.c = VSDController(m, GAINS)
        for s in ARM_SIDES:
            self.c.arms[s].drive = drives[s]
        self.c.hold_fingers(d.qpos)
        self.c.set_targets(targets)

    def compute(self, d):
        return self.c.compute(d)


class TipVSD:
    def __init__(self, m, d, tip_targets, drives):
        self.m, self.drives = m, drives
        n = len(FINGERS)
        self.k, self.c, self.fmax = GAINS.k / n, GAINS.zeta * np.sqrt(GAINS.k) / n, GAINS.f_max / n
        self.g = np.zeros(m.nv)
        self.sides = {}
        for s in ARM_SIDES:
            pts = {f: PoEPoint(m, f"{s}_{f}_tip", (0, 0, TIP_OFFSET[f])) for f in FINGERS}
            hj = hand_joint_ids(m, s)
            dofs = np.concatenate([arm_dofs(m, s), m.jnt_dofadr[hj]])
            HM = inertia_bound(m, dofs)
            C0 = np.diag(GAINS.alpha * np.sqrt(HM.sum(axis=1)))
            col = {dof: i for i, dof in enumerate(dofs)}
            fin_act = np.array([next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == j) for j in hj])
            self.sides[s] = dict(pts=pts, dofs=dofs, C0=C0, col=col, fin_act=fin_act,
                                 targets={f: tip_targets[s][f].copy() for f in FINGERS})
        self.info = {}

    def compute(self, d):
        m = self.m
        gravity(m, d, self.g)
        ctrl = np.zeros(m.nu)
        for s, S in self.sides.items():
            dofs = S["dofs"]
            qd = d.qvel[dofs]
            tau = -S["C0"] @ qd + self.g[dofs]
            for f, P in S["pts"].items():
                p, Jp = P.fk_jacobian(d.qpos)
                J = np.zeros((3, len(dofs)))
                J[:, [S["col"][x] for x in P.dofs]] = Jp
                force = _saturate(self.k * (p - S["targets"][f]), self.fmax) + self.c * (J @ qd)
                tau -= J.T @ force
            dr = self.drives[s]
            ctrl[dr.acts], _, _ = dr.command(d, tau[:7])
            ctrl[S["fin_act"]] = np.clip(tau[7:], -FINGER_TAU_MAX, FINGER_TAU_MAX)
        return ctrl


def classify(name):
    for f in FINGERS:
        if f"_{f}_" in name:
            return f
    return "palm" if "palm" in name else "arm"


def measure(m, d, side, desk, hand_bodies, lim_rows_jnt):
    f6 = np.zeros(6)
    forces = {k: 0.0 for k in (*FINGERS, "palm", "arm")}
    for i in range(d.ncon):
        c = d.contact[i]
        if desk not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == desk else c.geom1
        bname = m.body(m.geom_bodyid[other]).name
        if not bname.startswith(side + "_") and not bname.endswith("_" + side[0]):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        f_world = c.frame.reshape(3, 3).T @ f6[:3]
        f_on_hand = f_world if c.geom1 == desk else -f_world
        forces[classify(bname)] += f_on_hand[2]
    tau_lim = 0.0
    for r in range(d.nefc):
        if d.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT and d.efc_id[r] in lim_rows_jnt:
            tau_lim = max(tau_lim, abs(d.efc_force[r]))
    f_int = max(np.linalg.norm(d.cfrc_int[b, 3:]) for b in hand_bodies)
    return forces, tau_lim, f_int


def approach_state(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    c = VSDController(m, GAINS)
    c.hold_fingers(d.qpos)
    app = {s: Target(np.array([0.45, 0.20 if s == "left" else -0.20, APPROACH_Z]),
                     Ry(np.deg2rad(TILT_DEG)) @ R_PALM_DOWN[s]
                     @ Rz(np.deg2rad(ROLL_DEG if s == "left" else -ROLL_DEG))) for s in ARM_SIDES}
    c.set_targets(app)
    for _ in range(int(T_APPROACH / m.opt.timestep)):
        d.ctrl[:] = c.compute(d)
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    drives = {s: c.arms[s].drive for s in ARM_SIDES}
    return d, app, drives, d.ncon


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tip_positions(m, d, side):
    out = {}
    for f in FINGERS:
        b = m.body(f"{side}_{f}_tip").id
        out[f] = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0, 0, TIP_OFFSET[f]])
    return out


class Recorder:
    def __init__(self, m, size=(560, 420)):
        self.w, self.h = size
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, self.w)
        m.vis.global_.offheight = max(m.vis.global_.offheight, self.h)
        m.vis.map.force = 0.015
        self.r = mujoco.Renderer(m, self.h, self.w)
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.60, 0.20, 0.79]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.70, 235, -12

    def frame(self, d):
        self.r.update_scene(d, self.cam, self.opt)
        return self.r.render().copy()

    def close(self):
        self.r.close()


def press(m, method, d0, app, drives0, fps=30, record=True):
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = d0.qpos, d0.qvel, d0.ctrl
    d.time = 0.0
    mujoco.mj_forward(m, d)
    drives = {}
    for s in ARM_SIDES:
        dr = KinovaTorqueActuators(m, drives0[s].dofs, drives0[s].acts, mode=GAINS.actuator_mode)
        dr.integ = drives0[s].integ.copy()
        drives[s] = dr
    down = np.array([0, 0, DZ])
    ee_t = {s: Target(app[s].pos + down, app[s].R) for s in ARM_SIDES}
    if method == "servo":
        ctrl = JointServo(m, d, ee_t, drives)
    elif method == "ee_vsd":
        ctrl = EEVSD(m, d, ee_t, drives)
    else:
        tips = {s: {f: p + down for f, p in tip_positions(m, d, s).items()} for s in ARM_SIDES}
        ctrl = TipVSD(m, d, tips, drives)

    side = "left"
    desk = m.geom("desk").id
    hand_bodies = [b for b in range(m.nbody)
                   if m.body(b).name.startswith("left_") and "palm" not in m.body(b).name]
    lim_jnt = set(hand_joint_ids(m, side))
    fin_act = [a for a in range(m.nu) if m.joint(m.actuator_trnid[a, 0]).name in
               {m.joint(j).name for j in lim_jnt}]
    arm_dof = drives[side].dofs
    rec = Recorder(m) if record else None
    frames, log = [], {k: [] for k in ("t", "f_total", "f_fingers", "f_palm", "f_arm", "tau_lim", "f_int",
                                       "tau_fin", "tau_arm_ratio", "tip_z", "ee_z", "fin_defl")}
    fin_qadr = np.array([m.jnt_qposadr[j] for j in sorted(lim_jnt)])
    q_fin0 = d.qpos[fin_qadr].copy()
    next_frame = 0.0
    for _ in range(int(T_PRESS / m.opt.timestep)):
        d.ctrl[:] = ctrl.compute(d)
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        forces, tau_lim, f_int = measure(m, d, side, desk, hand_bodies, lim_jnt)
        log["t"].append(d.time)
        log["f_fingers"].append([forces[f] for f in FINGERS])
        log["f_palm"].append(forces["palm"])
        log["f_arm"].append(forces["arm"])
        log["f_total"].append(sum(forces.values()))
        log["tau_lim"].append(tau_lim)
        log["f_int"].append(f_int)
        log["tau_fin"].append(np.abs(d.actuator_force[fin_act]).max())
        log["tau_arm_ratio"].append(np.max(np.abs(drives[side].tau_m) / drives[side].hi))
        log["tip_z"].append([p[2] for p in tip_positions(m, d, side).values()])
        log["ee_z"].append(d.site_xpos[m.site("ee_site_l").id][2])
        log["fin_defl"].append(np.abs(d.qpos[fin_qadr] - q_fin0).max())
        if rec is not None and d.time >= next_frame - 1e-9:
            frames.append(rec.frame(d))
            next_frame += 1.0 / fps
    if rec is not None:
        rec.close()
    L = {k: np.array(v) for k, v in log.items()}
    np.savez_compressed(os.path.join(OUT, f"log_{method}.npz"), **L)
    return L, frames


def metrics(L):
    t = L["t"]
    tail = t >= t[-1] - 0.5
    fin = L["f_fingers"]
    return {
        "f_total_peak_N": float(L["f_total"].max()), "f_total_ss_N": float(L["f_total"][tail].mean()),
        "f_fingers_ss_N": fin[tail].mean(axis=0).tolist(),
        "f_palm_ss_N": float(L["f_palm"][tail].mean()), "f_arm_ss_N": float(L["f_arm"][tail].mean()),
        "tau_lim_peak_Nm": float(L["tau_lim"].max()), "tau_lim_ss_Nm": float(L["tau_lim"][tail].mean()),
        "f_int_peak_N": float(L["f_int"].max()), "f_int_ss_N": float(L["f_int"][tail].mean()),
        "tau_fin_peak_ratio": float(L["tau_fin"].max() / FINGER_TAU_MAX),
        "tau_arm_peak_ratio": float(L["tau_arm_ratio"].max()),
        "fin_defl_peak_deg": float(np.rad2deg(L["fin_defl"].max())),
        "fin_defl_ss_deg": float(np.rad2deg(L["fin_defl"][tail].mean())),
    }


def figures(R):
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color=SLOT[k], lw=2, label=v) for k, v in METHODS.items()]

    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for k in METHODS:
        L = R[k]
        axs[0].plot(L["t"], L["f_total"], color=SLOT[k])
        axs[1].plot(L["t"], L["tau_arm_ratio"], color=SLOT[k])
    axs[1].axhline(1.0, color=av.INK2, lw=1, ls=(0, (1, 2)))
    axs[0].set_yscale("symlog", linthresh=10)
    axs[0].set_ylabel("Total desk contact force [N]")
    axs[1].set_ylabel(r"Arm joint torque $\max_i|\tau_i|/\tau_{\max,i}$")
    axs[1].set_xlabel("Time after press starts [s]")
    axs[0].set_title("Contact force for a target below the desk (left hand)")
    axs[0].legend(handles=leg, loc="center right", fontsize=9)
    fig.savefig(os.path.join(OUT, "contact_force.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for k in METHODS:
        L = R[k]
        axs[0].plot(L["t"], L["tau_lim"], color=SLOT[k])
        axs[1].plot(L["t"], L["f_int"], color=SLOT[k])
        axs[2].plot(L["t"], L["tau_fin"], color=SLOT[k])
    axs[2].axhline(FINGER_TAU_MAX, color=av.INK2, lw=1, ls=(0, (1, 2)))
    axs[2].text(R["servo"]["t"][-1], FINGER_TAU_MAX, " actuator limit", color=av.INK2, va="bottom", ha="right", fontsize=8)
    axs[0].set_ylabel("Max joint-limit\nreaction [Nm]")
    axs[1].set_ylabel("Max finger link\ninternal force [N]")
    axs[2].set_ylabel("Max finger actuator\ntorque [Nm]")
    axs[2].set_xlabel("Time after press starts [s]")
    axs[0].set_title("Loads on finger joints and links (left hand, max over 16 joints)")
    axs[0].legend(handles=leg, loc="upper right", fontsize=9)
    fig.savefig(os.path.join(OUT, "finger_load.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = ["Index (ff)", "Middle (mf)", "Ring (rf)", "Thumb (th)"]
    x = np.arange(len(labels))
    wbar = 0.26
    for i, k in enumerate(METHODS):
        L = R[k]
        tail = L["t"] >= L["t"][-1] - 0.5
        vals = list(L["f_fingers"][tail].mean(axis=0))
        ax.bar(x + (i - 1) * (wbar + 0.02), vals, wbar, color=SLOT[k], label=METHODS[k])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("symlog", linthresh=5)
    ax.set_ylabel("Steady-state contact force [N]")
    ax.set_title("Steady-state contact force per finger (mean of last 0.5 s, no palm/arm contact)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(OUT, "contact_distribution.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_video(R, F, fps=30):
    font = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 16)
    n = min(len(F[k]) for k in METHODS)
    h, w = F["servo"][0].shape[:2]
    out = os.path.join(OUT, "contact_compare.mp4")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{3 * w + 16}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "20", out], stdin=subprocess.PIPE)
    snaps = {}
    for i in range(n):
        canvas = Image.new("RGB", (3 * w + 16, h), (252, 252, 251))
        t = i / fps
        for j, k in enumerate(METHODS):
            im = Image.fromarray(F[k][i])
            dr = ImageDraw.Draw(im)
            L = R[k]
            idx = min(np.searchsorted(L["t"], t), len(L["t"]) - 1)
            dr.rectangle((0, 0, w, 86), fill=(20, 30, 45))
            dr.text((10, 6), METHODS[k], font=font, fill=(255, 255, 255))
            dr.text((10, 34), f"Desk contact force {L['f_total'][idx]:7.1f} N", font=small, fill=(255, 255, 255))
            dr.text((10, 58), f"Finger joint-limit reaction {L['tau_lim'][idx]:5.2f} Nm", font=small, fill=(255, 220, 120))
            canvas.paste(im, (j * (w + 8), 0))
        ImageDraw.Draw(canvas).text((3 * w + 16 - 110, h - 26), f"t = {t:4.2f} s", font=small, fill=(255, 255, 255))
        proc.stdin.write(np.asarray(canvas).tobytes())
        if i in (0, n - 1):
            snaps[i] = canvas
    proc.stdin.close()
    proc.wait()
    snaps[n - 1].save(os.path.join(OUT, "snap_contact_final.png"))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d0, app, drives0, ncon0 = approach_state(m)
    tips0 = tip_positions(m, d0, "left")
    print("approach: contacts", ncon0, " lowest tip clearance [mm]",
          round((min(p[2] for p in tips0.values()) - 0.012 - DESK_TOP) * 1e3, 1))
    R, F, M = {}, {}, {}
    for k in METHODS:
        R[k], F[k] = press(m, k, d0, app, drives0)
        M[k] = metrics(R[k])
        print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else [round(x, 2) for x in b])
                             for a, b in M[k].items()}, ensure_ascii=False))
    from lie_jacobian import verify
    pts = [PoEPoint(m, f"{s}_{f}_tip", (0, 0, TIP_OFFSET[f])) for s in ARM_SIDES for f in FINGERS]
    M["poe_verify"] = verify(m, pts, n=100)
    M["setup"] = {"approach_z": APPROACH_Z, "tilt_deg": TILT_DEG, "roll_deg": ROLL_DEG, "dz": DZ, "approach_contacts": int(ncon0),
                  "approach_clearance_mm": float((min(p[2] for p in tips0.values()) - 0.012 - DESK_TOP) * 1e3),
                  "servo_kp": SERVO_KP.tolist(), "servo_kd": SERVO_KD.tolist(),
                  "tip_k": GAINS.k / len(FINGERS), "tip_c": GAINS.zeta * np.sqrt(GAINS.k) / len(FINGERS),
                  "tip_fmax": GAINS.f_max / len(FINGERS)}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(M, f, indent=2, ensure_ascii=False)
    figures(R)
    print("video ->", write_video(R, F))


if __name__ == "__main__":
    main()
