import argparse
import dataclasses
import json
import os
import subprocess
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ARM_SIDES, OUTPUT_DIR, SCENE_XML
from vsd_controller import Target, VSDController, VSDGains

GAINS = VSDGains(alpha=1.0, k=600.0, zeta=1.5, k_o=30.0, zeta_o=1.0,
                 f_max=15.0, m_max=3.0, k_rep=3.75, r_th=0.3,
                 actuator_mode="torque", use_friction_comp=False)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def make_schedule(name, home):
    if name == "reach":
        reach = {s: Target(np.array([0.55, 0.17 if s == "left" else -0.17, 0.95]),
                           Ry(np.deg2rad(30)) @ home[s].R) for s in ARM_SIDES}
        return [(0.0, "hold home", home), (0.5, "reach", reach), (4.0, "return home", home)], 7.0
    if name == "repulsion":
        cross = {s: Target(np.array([0.50, -0.05 if s == "left" else 0.05, 1.00]), home[s].R)
                 for s in ARM_SIDES}
        return [(0.0, "hold home", home), (0.5, "crossing targets", cross),
                (4.0, "return home", home)], 7.0
    raise ValueError(name)


class Logger:
    SIGNALS = ("p", "pd", "pdot", "e_o", "w", "q", "qd", "tau", "tau_m", "tau_s",
               "tau_damp", "tau_spring", "tau_tdamp", "tau_ori", "tau_g", "tau_fric", "tau_rep")

    def __init__(self):
        self.rows = {}

    def add(self, t, phase, info, ncon):
        row = {"t": t, "phase": phase, "ncon": ncon, "dist": info["dist"], "f_rep": info["f_rep"]}
        for s in ARM_SIDES:
            for k in self.SIGNALS:
                row[f"{s}_{k}"] = info[s][k]
        for k, v in row.items():
            self.rows.setdefault(k, []).append(np.copy(v))

    def save(self, path):
        np.savez_compressed(path, **{k: np.array(v) for k, v in self.rows.items()})


class VideoWriter:
    def __init__(self, m, path, fps=30, size=(720, 540)):
        self.m, self.fps = m, fps
        self.w, self.h = size
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, self.w)
        m.vis.global_.offheight = max(m.vis.global_.offheight, self.h)
        self.renderer = mujoco.Renderer(m, self.h, self.w, max_geom=20000)
        self.cams = []
        for az, el, dist, look in [(150, -20, 1.9, (0.35, 0, 1.0)), (180, -89.9, 1.5, (0.35, 0, 1.0))]:
            cam = mujoco.MjvCamera()
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            cam.lookat[:] = look
            self.cams.append(cam)
        self.font = ImageFont.truetype(FONT, 18)
        self.font_s = ImageFont.truetype(FONT, 15)
        self.trail = {s: [] for s in ARM_SIDES}
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{2 * self.w}x{self.h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
            stdin=subprocess.PIPE)

    def _add_geom(self, scn, gtype, size, pos, rgba, mat=np.eye(3)):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float),
                            np.asarray(pos, float), np.asarray(mat, float).ravel(),
                            np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def _decorate(self, scn, ctrl):
        for s in ARM_SIDES:
            tgt = ctrl.targets[s]
            self._add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0], tgt.pos, [0.1, 0.9, 0.2, 0.6])
            if tgt.R is not None:
                z = tgt.R[:, 2]
                g = scn.geoms[scn.ngeom] if scn.ngeom < scn.maxgeom else None
                if g is not None:
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                        np.eye(3).ravel(), np.array([0.1, 0.9, 0.2, 0.6], np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004,
                                         tgt.pos, tgt.pos + 0.08 * z)
                    scn.ngeom += 1
            pts = self.trail[s]
            for a, b in zip(pts[:-1], pts[1:]):
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                    np.eye(3).ravel(), np.array([1.0, 0.55, 0.0, 0.9], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, a, b)
                scn.ngeom += 1

    def frame(self, d, ctrl, phase, title):
        for s in ARM_SIDES:
            self.trail[s].append(ctrl.info[s]["p"].copy())
        views = []
        for cam in self.cams:
            self.renderer.update_scene(d, cam)
            self._decorate(self.renderer.scene, ctrl)
            views.append(self.renderer.render())
        img = Image.fromarray(np.concatenate(views, axis=1))
        draw = ImageDraw.Draw(img)
        draw.text((12, 10), f"{title}   t = {d.time:5.2f} s   [{phase}]", font=self.font, fill=(255, 255, 255))
        y = 36
        for s in ARM_SIDES:
            inf = ctrl.info[s]
            ep = np.linalg.norm(inf["p"] - inf["pd"]) * 1e3
            eo = np.rad2deg(np.linalg.norm(inf["e_o"]))
            qd = np.max(np.abs(inf["qd"]))
            draw.text((12, y), f"{s:5s}: |dp| = {ep:6.1f} mm   |e_o| = {eo:5.2f} deg   max|qd| = {qd:4.2f} rad/s",
                      font=self.font_s, fill=(255, 255, 255))
            y += 20
        draw.text((12, y), f"EE distance = {ctrl.info['dist']:.3f} m"
                           + ("   (repulsion active)" if np.any(ctrl.info["f_rep"]) else ""),
                  font=self.font_s, fill=(255, 220, 120) if np.any(ctrl.info["f_rep"]) else (255, 255, 255))
        draw.text((self.w + 12, self.h - 26), "top view", font=self.font_s, fill=(255, 255, 255))
        self.proc.stdin.write(np.asarray(img).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        self.renderer.close()


def run(scenario, gains, video=True, viewer=False, tag=None):
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    ctrl = VSDController(m, gains)
    ctrl.hold_fingers(d.qpos)
    home = {s: Target(d.site_xpos[a.site].copy(), d.site_xmat[a.site].reshape(3, 3).copy())
            for s, a in ctrl.arms.items()}
    schedule, duration = make_schedule(scenario, home)

    out = os.path.join(OUTPUT_DIR, scenario if not tag else f"{scenario}__{tag}")
    os.makedirs(out, exist_ok=True)
    log = Logger()
    vid = VideoWriter(m, os.path.join(out, f"vsd_{scenario}.mp4")) if video and not viewer else None
    title = f"VSD {scenario}  (Kinova Gen3 {gains.actuator_mode} mode, gear 100:1)"

    def step():
        phase, tgt = [(lbl, x) for t0, lbl, x in schedule if d.time >= t0 - 1e-9][-1]
        ctrl.set_targets(tgt)
        d.ctrl[:] = ctrl.compute(d)
        log.add(d.time, phase, ctrl.info, d.ncon)
        mujoco.mj_step(m, d)
        return phase

    if viewer:
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running() and d.time < duration:
                t0 = time.time()
                step()
                v.sync()
                time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))
    else:
        n_steps = int(round(duration / m.opt.timestep))
        next_frame = 0.0
        for _ in range(n_steps):
            t_now = d.time
            phase = step()
            if vid is not None and t_now >= next_frame - 1e-9:
                vid.frame(d, ctrl, phase, title)
                next_frame += 1.0 / vid.fps
    if vid is not None:
        vid.close()

    log.save(os.path.join(out, "log.npz"))
    meta = {
        "scenario": scenario,
        "tag": tag,
        "schedule": [{"t_start": t0, "phase": lbl,
                      **{f"{s}_pd": tg[s].pos.tolist() for s in ARM_SIDES},
                      **{f"{s}_Rd": tg[s].R.tolist() for s in ARM_SIDES}} for t0, lbl, tg in schedule],
        "duration": duration,
        "timestep": m.opt.timestep,
        "gains": dataclasses.asdict(gains),
        "arms": {s: {"C0_diag": np.diag(a.C0).tolist(), "gear": a.gear.tolist(),
                     "tau_max": a.tau_max.tolist(), "armature": m.dof_armature[a.dof].tolist(),
                     "damping": a.cv.tolist(), "frictionloss": a.cc.tolist(),
                     "qd_max": [1.39] * 4 + [1.22] * 3,
                     "q_range": m.jnt_range[m.dof_jntid[a.dof]].tolist(),
                     "q_limited": m.jnt_limited[m.dof_jntid[a.dof]].astype(bool).tolist()}
                 for s, a in ctrl.arms.items()},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    final = {s: (np.linalg.norm(ctrl.info[s]["p"] - ctrl.info[s]["pd"]) * 1e3,
                 np.rad2deg(np.linalg.norm(ctrl.info[s]["e_o"]))) for s in ARM_SIDES}
    print(f"[{scenario}] saved to {out}")
    for s, (ep, eo) in final.items():
        print(f"  {s}: final |dp| = {ep:.2f} mm, |e_o| = {eo:.3f} deg")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["reach", "repulsion", "all"], default="all")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--actuator-mode", choices=["torque", "current"], default=None,
                    help="current: open-loop motor torque (enables model friction compensation)")
    ap.add_argument("--friction-comp", choices=["on", "off"], default=None,
                    help="override controller friction compensation (tanh + viscous)")
    ap.add_argument("--no-saturation", action="store_true",
                    help="linear virtual springs, original VSD law (no f_max / m_max)")
    ap.add_argument("--damping", choices=["inertia", "uniform"], default=None,
                    help="joint damping design C0 (inertia-based vs uniform gains)")
    ap.add_argument("--tag", default=None, help="output folder suffix: output/<scenario>__<tag>")
    args = ap.parse_args()

    gains = dataclasses.replace(GAINS)
    if args.actuator_mode == "current":
        gains = dataclasses.replace(gains, actuator_mode="current", use_friction_comp=True)
    if args.friction_comp is not None:
        gains = dataclasses.replace(gains, use_friction_comp=args.friction_comp == "on")
    if args.no_saturation:
        gains = dataclasses.replace(gains, f_max=None, m_max=None)
    if args.damping is not None:
        gains = dataclasses.replace(gains, damping_mode=args.damping)
    for sc in (["reach", "repulsion"] if args.scenario == "all" else [args.scenario]):
        run(sc, gains, video=not args.no_video, viewer=args.viewer, tag=args.tag)


if __name__ == "__main__":
    main()
