import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ARM_SIDES, OUTPUT_DIR

COLOR = {"left": "#2a78d6", "right": "#eb6834"}
NAME = {"left": "Left arm", "right": "Right arm"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False,
})
AX = ["x", "y", "z"]


def load(scenario):
    d = os.path.join(OUTPUT_DIR, scenario)
    L = dict(np.load(os.path.join(d, "log.npz"), allow_pickle=False))
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return d, L, meta


def segments(L, meta):
    t = L["t"]
    sch = meta["schedule"]
    out = []
    for k, s in enumerate(sch):
        t0 = s["t_start"]
        t1 = sch[k + 1]["t_start"] if k + 1 < len(sch) else t[-1] + 1
        idx = np.where((t >= t0 - 1e-6) & (t < t1 - 1e-6))[0]
        if len(idx):
            out.append((s["phase"], idx[0], idx[-1] + 1))
    return out


def phase_lines(ax, meta):
    for s in meta["schedule"][1:]:
        ax.axvline(s["t_start"], color=MUTED, lw=1, ls=(0, (2, 2)), zorder=0)


def legend_arms(ax, extra=(), loc="upper right"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[s], lw=2, label=NAME[s]) for s in ARM_SIDES]
    for lbl, kw in extra:
        h.append(Line2D([], [], label=lbl, **kw))
    ax.legend(handles=h, loc=loc, fontsize=9, ncol=len(h) if loc == "upper right" else 1)


def step_metrics(t, p, pd, eo, qd, tau, qd_max, tau_max):
    e = p - pd
    e0 = e[0]
    n0 = np.linalg.norm(e0)
    en = np.linalg.norm(e, axis=1)
    tt = t - t[0]
    m = {"step_size_mm": n0 * 1e3}
    if n0 < 1e-4:
        return m
    u = e0 / n0
    proj = e @ u
    r = 1 - proj / n0

    def first(cond):
        i = np.argmax(cond)
        return float(tt[i]) if cond.any() else None

    t10, t90 = first(r >= 0.1), first(r >= 0.9)
    m["rise_time_s"] = (t90 - t10) if (t10 is not None and t90 is not None) else None
    band = max(0.02 * n0, 1e-3)
    out = np.where(en > band)[0]
    m["settling_time_2pct_s"] = float(tt[out[-1] + 1]) if len(out) and out[-1] + 1 < len(tt) else (0.0 if not len(out) else None)
    m["overshoot_pct"] = float(max(0.0, -proj.min()) / n0 * 100)
    tail = tt >= tt[-1] - 0.2
    m["steady_state_error_mm"] = float(en[tail].mean() * 1e3)
    m["steady_state_ori_error_deg"] = float(np.rad2deg(np.linalg.norm(eo[tail], axis=1).mean()))
    m["ori_step_deg"] = float(np.rad2deg(np.linalg.norm(eo[0])))
    v = np.linalg.norm(np.gradient(p, t, axis=0), axis=1)
    m["peak_ee_speed_mps"] = float(v.max())
    m["t_peak_speed_s"] = float(tt[v.argmax()])
    a, b = p[0], pd[0]
    ab = (b - a) / np.linalg.norm(b - a)
    dev = np.linalg.norm((p - a) - np.outer((p - a) @ ab, ab), axis=1)
    m["max_path_deviation_mm"] = float(dev.max() * 1e3)
    m["path_length_ratio"] = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) / np.linalg.norm(b - a))
    m["peak_qd_ratio"] = float((np.abs(qd).max(0) / qd_max).max())
    m["peak_tau_ratio"] = float((np.abs(tau).max(0) / tau_max).max())
    return m


def analyze(scenario):
    d, L, meta = load(scenario)
    out = os.path.join(d, "analysis")
    os.makedirs(out, exist_ok=True)
    t = L["t"]
    segs = segments(L, meta)
    arms = meta["arms"]
    figs = []

    def save(fig, name):
        fig.savefig(os.path.join(out, name), dpi=130, bbox_inches="tight")
        plt.close(fig)
        figs.append(name)

    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, ax in enumerate(axs):
        for s in ARM_SIDES:
            ax.plot(t, L[f"{s}_p"][:, i], color=COLOR[s])
            ax.plot(t, L[f"{s}_pd"][:, i], color=COLOR[s], lw=1.2, ls=(0, (4, 3)), alpha=0.8)
        ax.set_ylabel(f"{AX[i]} [m]")
        phase_lines(ax, meta)
    legend_arms(axs[0], [("Target $p_d$", dict(color=MUTED, lw=1.2, ls=(0, (4, 3))))])
    axs[0].set_title("End-effector position")
    axs[-1].set_xlabel("Time [s]")
    save(fig, "01_ee_position.png")

    fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for s in ARM_SIDES:
        axs[0].semilogy(t, np.maximum(np.linalg.norm(L[f"{s}_p"] - L[f"{s}_pd"], axis=1) * 1e3, 1e-3), color=COLOR[s])
        axs[1].plot(t, np.rad2deg(np.linalg.norm(L[f"{s}_e_o"], axis=1)), color=COLOR[s])
    axs[0].axhline(1.0, color=MUTED, lw=1, ls=(0, (1, 2)))
    axs[0].text(t[-1], 1.0, " 1 mm", color=INK2, va="center", fontsize=8)
    axs[0].set_ylabel("|Δp| [mm] (log)")
    axs[1].set_ylabel("|e_o| [deg]")
    axs[0].set_title("Position error / orientation error")
    axs[1].set_xlabel("Time [s]")
    for ax in axs:
        phase_lines(ax, meta)
    legend_arms(axs[0])
    save(fig, "02_error_norm.png")

    fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for s in ARM_SIDES:
        v = np.linalg.norm(L[f"{s}_pdot"], axis=1)
        acc = np.linalg.norm(np.gradient(L[f"{s}_pdot"], t, axis=0), axis=1)
        k = 25
        acc = np.convolve(acc, np.ones(k) / k, mode="same")
        axs[0].plot(t, v, color=COLOR[s])
        axs[1].plot(t, acc, color=COLOR[s])
    axs[0].set_ylabel(r"$|\dot{p}|$ [m/s]")
    axs[1].set_ylabel(r"$|\ddot{p}|$ [m/s²]")
    axs[0].set_title("End-effector speed / acceleration")
    axs[1].set_xlabel("Time [s]")
    for ax in axs:
        phase_lines(ax, meta)
    legend_arms(axs[0])
    save(fig, "03_ee_speed_accel.png")

    def joint_grid(key, ylabel, title, limit_fn, fname):
        fig, axs = plt.subplots(4, 2, figsize=(10, 9), sharex=True)
        axs = axs.ravel()
        for j in range(7):
            ax = axs[j]
            for s in ARM_SIDES:
                ax.plot(t, L[f"{s}_{key}"][:, j], color=COLOR[s], lw=1.6)
            lim = limit_fn(j)
            if lim is not None:
                for v in lim:
                    ax.axhline(v, color=INK2, lw=1, ls=(0, (1, 2)))
            ax.set_title(f"Joint {j + 1}", fontsize=10)
            phase_lines(ax, meta)
        axs[6].set_xlabel("Time [s]")
        axs[5].set_xlabel("Time [s]")
        axs[5].tick_params(labelbottom=True)
        axs[7].axis("off")
        legend_arms(axs[7], [("Limit", dict(color=INK2, lw=1, ls=(0, (1, 2))))], loc="center")
        fig.supylabel(ylabel, color=INK2)
        fig.suptitle(title, color=INK)
        fig.tight_layout()
        save(fig, fname)

    a0 = arms["left"]
    joint_grid("q", "q [rad]", "Joint angles",
               lambda j: a0["q_range"][j] if a0["q_limited"][j] else None, "04_joint_angle.png")
    joint_grid("qd", r"$\dot{q}$ [rad/s]", "Joint velocities (Gen3 limits: large 1.39 / small 1.22 rad/s)",
               lambda j: (-a0["qd_max"][j], a0["qd_max"][j]), "05_joint_velocity.png")
    joint_grid("tau_m", "τ [Nm]", "Joint torque commands (Gen3 limits: large 39 / small 9 Nm)",
               lambda j: (-a0["tau_max"][j], a0["tau_max"][j]), "06_joint_torque.png")

    step_segs = [sg for sg in segs if sg[0] != "hold home"]
    fig, axs = plt.subplots(len(step_segs), 3, figsize=(11, 3.4 * len(step_segs)), squeeze=False)
    for r, (lbl, i0, i1) in enumerate(step_segs):
        for c in range(3):
            ax = axs[r, c]
            for s in ARM_SIDES:
                e = (L[f"{s}_p"][i0:i1, c] - L[f"{s}_pd"][i0:i1, c]) * 1e3
                v = L[f"{s}_pdot"][i0:i1, c]
                ax.plot(e, v, color=COLOR[s], lw=1.6)
                ax.plot(e[0], v[0], "o", ms=8, mfc="none", mec=COLOR[s], mew=2)
            ax.plot(0, 0, "x", color=INK, ms=8, mew=2)
            ax.set_xlabel(rf"$\Delta p_{AX[c]}$ [mm]")
            ax.set_ylabel(rf"$\dot{{p}}_{AX[c]}$ [m/s]")
            ax.set_title(f"[{lbl}] {AX[c]} axis", fontsize=10)
    legend_arms(axs[0, 0], [("Start", dict(marker="o", ls="", mfc="none", mec=INK2, ms=8)),
                            ("Equilibrium", dict(marker="x", ls="", color=INK, ms=8))], loc="lower left")
    fig.suptitle(r"Task-space phase portrait ($\Delta p_i$ vs $\dot{p}_i$)", color=INK)
    fig.tight_layout()
    save(fig, "07_phase_task.png")

    fig, axs = plt.subplots(len(step_segs), 7, figsize=(17, 3.0 * len(step_segs)), squeeze=False)
    for r, (lbl, i0, i1) in enumerate(step_segs):
        for j in range(7):
            ax = axs[r, j]
            for s in ARM_SIDES:
                q = L[f"{s}_q"][i0:i1, j]
                qd = L[f"{s}_qd"][i0:i1, j]
                ax.plot(q - q[-1], qd, color=COLOR[s], lw=1.4)
                ax.plot(q[0] - q[-1], qd[0], "o", ms=7, mfc="none", mec=COLOR[s], mew=1.8)
            ax.plot(0, 0, "x", color=INK, ms=7, mew=2)
            ax.set_title(f"[{lbl}] joint {j + 1}", fontsize=9)
            ax.set_xlabel(r"$q - q_{eq}$ [rad]", fontsize=8)
            if j == 0:
                ax.set_ylabel(r"$\dot{q}$ [rad/s]")
            ax.tick_params(labelsize=8)
    legend_arms(axs[0, 0], loc="lower left")
    fig.suptitle(r"Joint-space phase portrait ($q_i - q_{eq}$ vs $\dot{q}_i$)", color=INK)
    fig.tight_layout()
    save(fig, "08_phase_joint.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for c, (i, j, ttl) in enumerate([(0, 1, "Top view (x–y)"), (0, 2, "Side view (x–z)")]):
        ax = axs[c]
        for s in ARM_SIDES:
            p = L[f"{s}_p"]
            ax.plot(p[:, i], p[:, j], color=COLOR[s], lw=1.8)
            for lbl, i0, i1 in step_segs:
                a, b = p[i0], L[f"{s}_pd"][i0]
                ax.plot([a[i], b[i]], [a[j], b[j]], color=MUTED, lw=1, ls=(0, (4, 3)))
                ax.plot(b[i], b[j], "x", color=INK, ms=7, mew=2)
        ax.set_xlabel(f"{AX[i]} [m]")
        ax.set_ylabel(f"{AX[j]} [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(ttl)
    legend_arms(axs[0], [("Start–target line", dict(color=MUTED, lw=1, ls=(0, (4, 3)))),
                         ("Target", dict(marker="x", ls="", color=INK, ms=7))], loc="best")
    fig.tight_layout()
    save(fig, "09_ee_path.png")

    g = meta["gains"]
    if np.any(L["f_rep"]) or scenario == "repulsion":
        fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
        axs[0].plot(t, L["dist"], color=INK)
        axs[0].axhline(g["r_th"], color=INK2, lw=1, ls=(0, (1, 2)))
        axs[0].text(t[-1], g["r_th"], f" r_th = {g['r_th']} m", color=INK2, va="bottom", fontsize=8, ha="right")
        axs[0].set_ylabel("‖x_r − x_l‖ [m]")
        axs[0].set_title("Distance between end-effectors / virtual repulsive force")
        axs[1].plot(t, np.linalg.norm(L["f_rep"], axis=1), color=INK)
        axs[1].set_ylabel("|f_rep| [N]")
        axs[1].set_xlabel("Time [s]")
        for ax in axs:
            phase_lines(ax, meta)
        save(fig, "10_dual_arm_distance.png")

    metrics = {"scenario": scenario, "segments": {}}
    for lbl, i0, i1 in segs:
        metrics["segments"][lbl] = {}
        for s in ARM_SIDES:
            metrics["segments"][lbl][s] = step_metrics(
                t[i0:i1], L[f"{s}_p"][i0:i1], L[f"{s}_pd"][i0:i1], L[f"{s}_e_o"][i0:i1],
                L[f"{s}_qd"][i0:i1], L[f"{s}_tau_m"][i0:i1],
                np.array(arms[s]["qd_max"]), np.array(arms[s]["tau_max"]))
    metrics["min_ee_distance_m"] = float(L["dist"].min())
    metrics["max_contacts"] = int(L["ncon"].max())
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_report(out, scenario, meta, metrics, figs)
    print(f"[{scenario}] analysis -> {out}")
    return metrics


def _f(v, fmt):
    return "–" if v is None else format(v, fmt)


def write_report(out, scenario, meta, metrics, figs):
    g = meta["gains"]
    a = meta["arms"]["left"]
    lines = [
        f"# VSD control analysis — `{scenario}`", "",
        "## Controller",
        "VSD control law (Arimoto & Sekimoto):",
        "",
        "```",
        "τ = −C₀q̇ − Jᵀ(k·Δp + ζ√k·ṗ) + Jωᵀ(k_o·e_o − ζ_o√k_o·ω) + g(q) + τ_fric + Jᵀf_rep",
        "C₀ = α·√H₀ ,  H₀ = diag(Σⱼ H_M,ij)",
        "```",
        "",
        "| Parameter | Value |", "|---|---|",
        f"| Task-space spring k | {g['k']} N/m (saturation {g['f_max']} N) |",
        f"| Task-space damping ζ | {g['zeta']} (c = ζ√k = {g['zeta'] * np.sqrt(g['k']):.1f} N·s/m) |",
        f"| Joint damping α | {g['alpha']} → C₀ = diag({', '.join(f'{c:.2f}' for c in a['C0_diag'])}) N·m·s/rad |",
        f"| Orientation spring k_o / ζ_o | {g['k_o']} N·m/rad (saturation {g['m_max']} N·m) / {g['zeta_o']} |",
        f"| Repulsion k_rep / r_th | {g['k_rep']} N·m / {g['r_th']} m |",
        f"| Actuator mode | {g['actuator_mode']} (friction compensation {'on' if g['use_friction_comp'] else 'off'}) |",
        "",
        "## Actuator model (Kinova Gen3)",
        "| Item | Large (1–4) | Small (5–7) |", "|---|---|---|",
        f"| Gear ratio | {a['gear'][0]:.0f}:1 | {a['gear'][6]:.0f}:1 |",
        f"| Torque limit | {a['tau_max'][0]:.0f} Nm | {a['tau_max'][6]:.0f} Nm |",
        f"| Velocity limit | {a['qd_max'][0]} rad/s | {a['qd_max'][6]} rad/s |",
        f"| Reflected rotor inertia | {a['armature'][0]:.4g} kg·m² | {a['armature'][6]:.4g} kg·m² |",
        f"| Friction (Coulomb / viscous) | {a['frictionloss'][0]} Nm / {a['damping'][0]} Nm·s/rad | {a['frictionloss'][6]} Nm / {a['damping'][6]} Nm·s/rad |",
        "",
        "## Step-response metrics", "",
        "| Segment | Arm | Step [mm] | Orientation step [deg] | Rise time 10–90% [s] | Settling time 2% [s] | Overshoot [%] | Steady-state error [mm] / [deg] | Peak EE speed [m/s] | Max path deviation [mm] | Path length / straight line | Peak q̇/limit | Peak τ/limit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lbl, per in metrics["segments"].items():
        for s, m in per.items():
            if "rise_time_s" not in m:
                lines.append(f"| {lbl} | {NAME[s]} | {m['step_size_mm']:.1f} | – | – | – | – | – | – | – | – | – | – |")
                continue
            lines.append(
                f"| {lbl} | {NAME[s]} | {m['step_size_mm']:.1f} | {m['ori_step_deg']:.1f} | {_f(m['rise_time_s'], '.2f')} | "
                f"{_f(m['settling_time_2pct_s'], '.2f')} | {m['overshoot_pct']:.1f} | "
                f"{m['steady_state_error_mm']:.2f} / {m['steady_state_ori_error_deg']:.3f} | "
                f"{m['peak_ee_speed_mps']:.3f} | {m['max_path_deviation_mm']:.1f} | {m['path_length_ratio']:.3f} | "
                f"{m['peak_qd_ratio']:.2f} | {m['peak_tau_ratio']:.2f} |")
    lines += ["",
              f"- Minimum distance between end-effectors: {metrics['min_ee_distance_m']:.3f} m (repulsion threshold r_th = {g['r_th']} m)",
              f"- Maximum number of contacts: {metrics['max_contacts']}",
              "", "## Figures", ""]
    lines += [f"![{f}]({f})" for f in figs]
    with open(os.path.join(out, "analysis_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    args = ap.parse_args()
    scs = [args.scenario] if args.scenario else sorted(
        x for x in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, x, "log.npz")))
    for sc in scs:
        analyze(sc)


if __name__ == "__main__":
    main()
