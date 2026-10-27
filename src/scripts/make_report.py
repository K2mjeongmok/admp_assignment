import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import OUTPUT_DIR, SRC_DIR

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(SRC_DIR, "report")
FIG = os.path.join(REPORT, "figures")
GEN = os.path.join(REPORT, "generated")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

ABLATIONS = [("current_fc", ["--actuator-mode", "current"]),
             ("current_nofc", ["--actuator-mode", "current", "--friction-comp", "off"]),
             ("linear_spring", ["--no-saturation"]),
             ("uniform_damping", ["--damping", "uniform"])]

FIGURES = {
    "reach_position.png": "reach/analysis/01_ee_position.png",
    "reach_error.png": "reach/analysis/02_error_norm.png",
    "reach_speed.png": "reach/analysis/03_ee_speed_accel.png",
    "reach_qd.png": "reach/analysis/05_joint_velocity.png",
    "reach_tau.png": "reach/analysis/06_joint_torque.png",
    "reach_phase_task.png": "reach/analysis/07_phase_task.png",
    "reach_phase_joint.png": "reach/analysis/08_phase_joint.png",
    "reach_path.png": "reach/analysis/09_ee_path.png",
    "rep_distance.png": "repulsion/analysis/10_dual_arm_distance.png",
    "rep_path.png": "repulsion/analysis/09_ee_path.png",
    "cmp_actuator.png": "comparison/cmp_actuator.png",
    "cmp_spring.png": "comparison/cmp_spring.png",
    "cmp_damping.png": "comparison/cmp_damping.png",
    "w3_gravity.png": "stages/w3_gravity.png",
    "w3_push.png": "stages/w3_push.png",
    "w5_stages.png": "stages/w5_stages.png",
    "ct_force.png": "contact/contact_force.png",
    "ct_finger_load.png": "contact/finger_load.png",
    "ct_distribution.png": "contact/contact_distribution.png",
    "ct_snap.png": "contact/snap_contact_final.png",
}


def sh(cmd, cwd=SCRIPTS):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_simulations():
    sh([sys.executable, "stage_experiments.py"])
    sh([sys.executable, "contact_experiment.py"])
    sh([sys.executable, "run_vsd.py"])
    for tag, extra in ABLATIONS:
        sh([sys.executable, "run_vsd.py", "--scenario", "reach", "--no-video", "--tag", tag, *extra])


def snapshot_strip(video, times, out_png, crop_top=105):
    font = ImageFont.truetype(FONT, 26)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, t in enumerate(times):
            png = os.path.join(tmp, f"{k}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", video,
                            "-frames:v", "1", png], check=True)
            im = Image.open(png).convert("RGB")
            w, h = im.size
            im = im.crop((0, crop_top, w // 2, h))
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, 150, 40), fill=(20, 30, 45))
            draw.text((10, 4), f"t = {t:.1f} s", font=font, fill=(255, 255, 255))
            frames.append(im)
    gap = 8
    W = sum(f.width for f in frames) + gap * (len(frames) - 1)
    strip = Image.new("RGB", (W, frames[0].height), (252, 252, 251))
    x = 0
    for f in frames:
        strip.paste(f, (x, 0))
        x += f.width + gap
    strip.save(out_png)


def full_frame(video, t, out_png, crop_top=105):
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "f.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", video,
                        "-frames:v", "1", png], check=True)
        im = Image.open(png).convert("RGB")
        im.crop((0, crop_top, im.width, im.height)).save(out_png)


def load_run(run):
    d = os.path.join(OUTPUT_DIR, run)
    meta = json.load(open(os.path.join(d, "meta.json")))
    met = json.load(open(os.path.join(d, "analysis", "metrics.json")))
    return meta, met


def f(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


def seg(met, name, side="left"):
    return met["segments"][name][side]


def write_numbers():
    meta, met = load_run("reach")
    g, a = meta["gains"], meta["arms"]["left"]
    _, rep = load_run("repulsion")
    r, rt = seg(met, "reach"), seg(met, "return home")
    rr = seg(rep, "return home")
    cfc = seg(load_run("reach__current_fc")[1], "reach")
    cno = seg(load_run("reach__current_nofc")[1], "reach")
    lin = seg(load_run("reach__linear_spring")[1], "reach")
    lin_r = seg(load_run("reach__linear_spring")[1], "return home")
    uni = seg(load_run("reach__uniform_damping")[1], "reach")
    c0 = a["C0_diag"]
    c0_uniform = load_run("reach__uniform_damping")[0]["arms"]["left"]["C0_diag"][0]
    nums = {
        "Gk": f"{g['k']:.0f}", "Gzeta": f"{g['zeta']:.1f}", "Galpha": f"{g['alpha']:.1f}",
        "Gc": f"{g['zeta'] * np.sqrt(g['k']):.1f}",
        "Gko": f"{g['k_o']:.0f}", "Gzetao": f"{g['zeta_o']:.1f}", "Gco": f"{g['zeta_o'] * np.sqrt(g['k_o']):.2f}",
        "Gfmax": f"{g['f_max']:.0f}", "Gmmax": f"{g['m_max']:.0f}",
        "Gkrep": f"{g['k_rep']:.2f}", "Grth": f"{g['r_th']:.2f}",
        "Gdeq": f"{g['k_rep'] / g['f_max']:.2f}",
        "CzeroList": ", ".join(f"{c:.2f}" for c in c0), "CzeroUniform": f"{c0_uniform:.2f}",
        "ArmLarge": f"{a['armature'][0]:.3f}", "ArmSmall": f"{a['armature'][6]:.4f}",
        "RStep": f(r["step_size_mm"], 0), "ROri": f(r["ori_step_deg"], 1),
        "RRise": f(r["rise_time_s"]), "RTs": f(r["settling_time_2pct_s"]), "ROS": f(r["overshoot_pct"], 1),
        "REss": f(r["steady_state_error_mm"]), "REssOri": f(r["steady_state_ori_error_deg"], 3),
        "RVmax": f(r["peak_ee_speed_mps"]), "RDev": f(r["max_path_deviation_mm"], 1),
        "RPath": f(r["path_length_ratio"]), "RQd": f(100 * r["peak_qd_ratio"], 0),
        "RTau": f(100 * r["peak_tau_ratio"], 0),
        "RtTs": f(rt["settling_time_2pct_s"]), "RtOS": f(rt["overshoot_pct"], 1),
        "RtQd": f(100 * rt["peak_qd_ratio"], 0), "RtTau": f(100 * rt["peak_tau_ratio"], 0),
        "PMinDist": f(rep["min_ee_distance_m"], 3), "PContacts": str(rep["max_contacts"]),
        "PRetTs": f(rr["settling_time_2pct_s"]),
        "CfcEss": f(cfc["steady_state_error_mm"], 1), "CfcEssOri": f(cfc["steady_state_ori_error_deg"], 1),
        "CnoEss": f(cno["steady_state_error_mm"], 1), "CnoEssOri": f(cno["steady_state_ori_error_deg"], 1),
        "LQd": f(lin["peak_qd_ratio"], 1), "LQdRet": f(lin_r["peak_qd_ratio"], 1),
        "LOS": f(lin["overshoot_pct"], 1), "LVmax": f(lin["peak_ee_speed_mps"]),
        "LTs": f(lin["settling_time_2pct_s"]), "LTau": f(100 * lin["peak_tau_ratio"], 0),
        "UTs": f(uni["settling_time_2pct_s"]), "UOS": f(uni["overshoot_pct"], 1),
        "UDev": f(uni["max_path_deviation_mm"], 1),
        "FricEst": f(a["frictionloss"][0] / (g["k"] * 0.5) * 1e3, 1),
    }
    nums.update(stage_numbers())
    nums.update(contact_numbers())
    L = np.load(os.path.join(OUTPUT_DIR, "reach", "log.npz"))
    e = np.concatenate([L[f"{sd}_tau"] - L[f"{sd}_tau_s"] for sd in ("left", "right")])
    nums["TrkRms"] = f"{np.sqrt(np.mean(e ** 2)):.3f}"
    nums["TrkMax"] = f"{np.abs(e).max():.2f}"
    with open(os.path.join(GEN, "numbers.tex"), "w") as fp:
        for k, v in nums.items():
            fp.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def sci(v):
    mant, ex = f"{v:.1e}".split("e")
    return f"{mant}\\times10^{{{int(ex)}}}"


def stage_numbers():
    V = json.load(open(os.path.join(OUTPUT_DIR, "stages", "verification.json")))
    k, gr, st = V["week2_kinematics"], V["week3_gravity"], V["stages"]
    mx = lambda key: max(k[s][key] for s in ("left", "right"))
    return {
        "VKinN": str(k["left"]["n_samples"]),
        "VKinPos": sci(mx("fk_pos_max_err_m")), "VKinRot": sci(mx("fk_rot_max_err")),
        "VKinJv": sci(mx("jv_max_err")), "VKinJw": sci(mx("jw_max_err")),
        "VGravN": str(gr["n_samples"]), "VGrav": sci(gr["max_abs_err_Nm"]), "VGravMax": f"{gr['max_abs_g_Nm']:.1f}",
        "ZeroDrop": f"{st['w3_zero']['z_drop_mm']:.0f}",
        "GHoldDrift": f"{st['w3_g_torque_hold']['max_drift_before_push_mm']:.1f}",
        "PushNoComp": f"{st['w3_g_current']['final_disp_mm']:.1f}",
        "PushNoCompStop": f"{st['w3_g_current']['stop_time_s']:.2f}",
        "PushComp": f"{st['w3_gf_current']['final_disp_mm']:.0f}",
        "GfDrift": f"{st['w3_gf_current']['drift_before_push_mm']:.0f}",
        "PushTorque": f"{st['w3_g_torque']['final_disp_mm']:.0f}",
        "SpringErr": f"{st['w5_spring']['err_tail_mean_mm']:.0f}",
        "SpringNull": f"{st['w5_spring']['null_speed_tail']:.2f}",
        "SpringDev": f"{st['w5_spring']['max_path_dev_mm']:.0f}",
        "SpringContacts": str(st["w5_spring"]["max_contacts"]),
        "COErr": f"{st['w5_spring_c0']['err_tail_mean_mm']:.2f}",
        "CONull": f"{st['w5_spring_c0']['null_speed_tail']:.3f}",
        "CODev": f"{st['w5_spring_c0']['max_path_dev_mm']:.0f}",
        "TaskErr": sci(st["w5_spring_c0_task"]["err_tail_mean_mm"]),
        "TaskDev": f"{st['w5_spring_c0_task']['max_path_dev_mm']:.1f}",
        "FullDev": f"{st['w5_full']['max_path_dev_mm']:.0f}",
    }


def contact_numbers():
    C = json.load(open(os.path.join(OUTPUT_DIR, "contact", "metrics.json")))
    sv, ee, tp, st, pv = C["servo"], C["ee_vsd"], C["tip_vsd"], C["setup"], C["poe_verify"]
    fl = lambda v, nd=1: f"{v:.{nd}f}"
    return {
        "CtClear": fl(st["approach_clearance_mm"], 0), "CtDz": fl(-st["dz"] * 100, 0),
        "CtDepth": fl(-st["dz"] * 1000 - st["approach_clearance_mm"], 0),
        "CtTilt": fl(st["tilt_deg"], 0), "CtRoll": fl(st["roll_deg"], 0),
        "CtTipK": fl(st["tip_k"], 0), "CtTipC": fl(st["tip_c"], 2), "CtTipF": fl(st["tip_fmax"], 2),
        "CtPoePos": sci(pv["pos_max_err_m"]), "CtPoeJac": sci(pv["jac_max_err"]), "CtPoeN": str(pv["n_samples"]),
        "CtSvPeak": fl(sv["f_total_peak_N"], 0), "CtSvSs": fl(sv["f_total_ss_N"], 0),
        "CtEePeak": fl(ee["f_total_peak_N"]), "CtEeSs": fl(ee["f_total_ss_N"]),
        "CtTpPeak": fl(tp["f_total_peak_N"]), "CtTpSs": fl(tp["f_total_ss_N"]),
        "CtRatio": fl(sv["f_total_ss_N"] / ee["f_total_ss_N"], 1),
        "CtRatioPeak": fl(sv["f_total_peak_N"] / ee["f_total_peak_N"], 1),
        "CtEeIntPk": fl(ee["f_int_peak_N"]), "CtTpIntPk": fl(tp["f_int_peak_N"]),
        "CtSvLim": fl(sv["tau_lim_peak_Nm"], 2), "CtSvInt": fl(sv["f_int_peak_N"], 0),
        "CtSvIntSs": fl(sv["f_int_ss_N"]), "CtEeInt": fl(ee["f_int_ss_N"]), "CtTpInt": fl(tp["f_int_ss_N"]),
        "CtIntRed": fl(100 * (1 - tp["f_int_ss_N"] / ee["f_int_ss_N"]), 0),
        "CtEeFin": fl(100 * ee["tau_fin_peak_ratio"], 0), "CtTpFin": fl(100 * tp["tau_fin_peak_ratio"], 0),
        "CtEeArm": fl(100 * ee["tau_arm_peak_ratio"], 0), "CtTpArm": fl(100 * tp["tau_arm_peak_ratio"], 0),
        "CtEeDefl": fl(ee["fin_defl_ss_deg"], 0),
        "CtEeFF": fl(ee["f_fingers_ss_N"][0]), "CtEeMF": fl(ee["f_fingers_ss_N"][1]),
        "CtEeRF": fl(ee["f_fingers_ss_N"][2]), "CtEeTH": fl(ee["f_fingers_ss_N"][3]),
        "CtTpMin": fl(min(tp["f_fingers_ss_N"])), "CtTpMax": fl(max(tp["f_fingers_ss_N"])),
        "CtSvFF": fl(sv["f_fingers_ss_N"][0], 0), "CtSvTH": fl(sv["f_fingers_ss_N"][3], 0),
    }


def write_contact_table():
    C = json.load(open(os.path.join(OUTPUT_DIR, "contact", "metrics.json")))
    names = {"servo": "Position tracking (joint servo)", "ee_vsd": "End-effector VSD", "tip_vsd": "Fingertip VSD"}
    lines = [r"\begin{tabular}{lrrcrrrr}", r"\toprule",
             r"Method & \multicolumn{2}{c}{Desk contact force [N]} & Per-finger steady state [N] & Joint-limit & Link internal & "
             r"Finger torque & Arm torque \\",
             r" & peak & steady & (index/middle/ring/thumb) & reaction [Nm] & force, steady [N] & peak [\%] & peak [\%] \\",
             r"\midrule"]
    for k, lbl in names.items():
        c = C[k]
        ff = " / ".join(f"{v:.1f}" for v in c["f_fingers_ss_N"])
        lines.append(f"{lbl} & {c['f_total_peak_N']:.1f} & {c['f_total_ss_N']:.1f} & {ff} & "
                     f"{c['tau_lim_peak_Nm']:.2f} & {c['f_int_ss_N']:.1f} & "
                     f"{100 * c['tau_fin_peak_ratio']:.0f} & {100 * c['tau_arm_peak_ratio']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GEN, "tab_contact.tex"), "w") as fp:
        fp.write("\n".join(lines) + "\n")


def write_tables():
    meta, met = load_run("reach")
    _, rep = load_run("repulsion")
    name = {"left": "Left", "right": "Right"}
    seg_ko = {"reach": "Reach", "return home": "Return home", "crossing targets": "Crossing targets"}

    def rows(m, segments):
        out = []
        for sname in segments:
            for side in ("left", "right"):
                s = m["segments"][sname][side]
                out.append(
                    f"{seg_ko[sname]} & {name[side]} & {f(s['step_size_mm'], 1)} & {f(s['ori_step_deg'], 1)} & "
                    f"{f(s.get('rise_time_s'))} & {f(s.get('settling_time_2pct_s'))} & {f(s['overshoot_pct'], 1)} & "
                    f"{f(s['steady_state_error_mm'])} / {f(s['steady_state_ori_error_deg'], 3)} & "
                    f"{f(s['max_path_deviation_mm'], 1)} & {f(100 * s['peak_qd_ratio'], 0)} & "
                    f"{f(100 * s['peak_tau_ratio'], 0)} \\\\")
        return "\n".join(out)

    head = (r"\begin{tabular}{llrrrrrrrrr}" "\n" r"\toprule" "\n"
            r"Segment & Arm & $\|\Delta p_0\|$ & $\|e_{o,0}\|$ & $t_r$ & $t_s$ & OS & $e_{ss}$ & Path dev. & "
            r"$\dot q/\dot q_{\max}$ & $\tau/\tau_{\max}$ \\" "\n"
            r" & & [mm] & [deg] & [s] & [s] & [\%] & [mm]/[deg] & [mm] & [\%] & [\%] \\" "\n" r"\midrule" "\n")
    with open(os.path.join(GEN, "tab_reach.tex"), "w") as fp:
        fp.write(head + rows(met, ["reach", "return home"]) + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(GEN, "tab_repulsion.tex"), "w") as fp:
        fp.write(head + rows(rep, ["crossing targets", "return home"]) + "\n\\bottomrule\n\\end{tabular}\n")

    variants = [("reach", r"Proposed: torque mode + saturated spring + inertia-based $C_0$"),
                ("reach__current_fc", r"Current mode + friction comp. $\hat\tau_{fric}$"),
                ("reach__current_nofc", "Current mode, no compensation"),
                ("reach__linear_spring", r"Linear spring (no saturation)"),
                ("reach__uniform_damping", r"Uniform $C_0$ (same trace)")]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Condition & $t_s$ [s] & OS [\%] & $e_{ss}$ [mm] & $e_{o,ss}$ [deg] & "
             r"$\dot q/\dot q_{\max}$ [\%] & $\tau/\tau_{\max}$ [\%] \\", r"\midrule"]
    for run, lbl in variants:
        _, m = load_run(run)
        s = m["segments"]["reach"]["left"]
        lines.append(f"{lbl} & {f(s.get('settling_time_2pct_s'))} & {f(s['overshoot_pct'], 1)} & "
                     f"{f(s['steady_state_error_mm'])} & {f(s['steady_state_ori_error_deg'], 2)} & "
                     f"{f(100 * s['peak_qd_ratio'], 0)} & {f(100 * s['peak_tau_ratio'], 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GEN, "tab_ablation.tex"), "w") as fp:
        fp.write("\n".join(lines) + "\n")


def build_assets():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(GEN, exist_ok=True)
    for dst, src in FIGURES.items():
        shutil.copy(os.path.join(OUTPUT_DIR, src), os.path.join(FIG, dst))
    snapshot_strip(os.path.join(OUTPUT_DIR, "reach", "vsd_reach.mp4"), [0.5, 0.9, 1.3, 2.5],
                   os.path.join(FIG, "snap_reach.png"))
    full_frame(os.path.join(OUTPUT_DIR, "repulsion", "vsd_repulsion.mp4"), 3.5,
               os.path.join(FIG, "snap_repulsion.png"))
    full_frame(os.path.join(OUTPUT_DIR, "reach", "vsd_reach.mp4"), 0.2,
               os.path.join(FIG, "snap_home.png"))
    write_numbers()
    write_tables()
    write_contact_table()


def compile_pdf():
    if shutil.which("latexmk") and shutil.which("xelatex"):
        sh(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=REPORT)
    elif shutil.which("xelatex"):
        for _ in range(2):
            sh(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=REPORT)
    elif shutil.which("tectonic"):
        sh(["tectonic", "main.tex"], cwd=REPORT)
    else:
        print("\n[!] XeLaTeX is not installed, so the PDF was not built. Install it and run again:\n"
              "    sudo apt install texlive-xetex texlive-latex-extra texlive-science "
              "texlive-fonts-recommended latexmk")
        return False
    print("PDF ->", os.path.join(REPORT, "main.pdf"))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="rerun all simulations")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    if args.rerun:
        run_simulations()
    sh([sys.executable, "analyze_vsd.py"])
    sh([sys.executable, "compare_runs.py"])
    build_assets()
    if not args.no_pdf:
        compile_pdf()


if __name__ == "__main__":
    main()
