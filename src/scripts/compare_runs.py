import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_vsd as av
from common import OUTPUT_DIR

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
SIDE = "left"

GROUPS = {
    "cmp_actuator": [("reach", "Torque mode (torque-sensor loop)"),
                     ("reach__current_fc", "Current mode + friction comp."),
                     ("reach__current_nofc", "Current mode (no comp.)")],
    "cmp_spring": [("reach", "Saturated spring ($f_{max}$, $m_{max}$)"),
                   ("reach__linear_spring", "Linear spring (no saturation)")],
    "cmp_damping": [("reach", r"Inertia-based $C_0$"),
                    ("reach__uniform_damping", r"Uniform $C_0$ (same trace)")],
}


def load(run):
    d = os.path.join(OUTPUT_DIR, run)
    L = dict(np.load(os.path.join(d, "log.npz")))
    meta = json.load(open(os.path.join(d, "meta.json")))
    return L, meta


def legend(ax, runs, loc="upper right"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=SLOT[i], lw=2, label=lbl) for i, (_, lbl) in enumerate(runs)]
    ax.legend(handles=h, loc=loc, fontsize=9)


def err_axes(axs, runs, data):
    for i, (run, lbl) in enumerate(runs):
        L, meta = data[run]
        t = L["t"]
        ep = np.linalg.norm(L[f"{SIDE}_p"] - L[f"{SIDE}_pd"], axis=1) * 1e3
        eo = np.rad2deg(np.linalg.norm(L[f"{SIDE}_e_o"], axis=1))
        axs[0].semilogy(t, np.maximum(ep, 1e-3), color=SLOT[i])
        axs[1].plot(t, eo, color=SLOT[i])
    axs[0].set_ylabel("|Δp| [mm] (log)")
    axs[1].set_ylabel("|e_o| [deg]")
    for ax in axs:
        av.phase_lines(ax, data[runs[0][0]][1])


def qd_ratio(L, meta):
    lim = np.array(meta["arms"][SIDE]["qd_max"])
    return np.max(np.abs(L[f"{SIDE}_qd"]) / lim, axis=1)


def main():
    out = os.path.join(OUTPUT_DIR, "comparison")
    os.makedirs(out, exist_ok=True)
    data = {run: load(run) for runs in GROUPS.values() for run, _ in runs
            if os.path.isfile(os.path.join(OUTPUT_DIR, run, "log.npz"))}

    runs = [r for r in GROUPS["cmp_actuator"] if r[0] in data]
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    err_axes(axs, runs, data)
    axs[0].set_title("Errors for each actuation mode (left arm)")
    axs[1].set_xlabel("Time [s]")
    legend(axs[0], runs, loc="lower left")
    fig.savefig(os.path.join(out, "cmp_actuator.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    runs = [r for r in GROUPS["cmp_spring"] if r[0] in data]
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for i, (run, lbl) in enumerate(runs):
        L, meta = data[run]
        axs[0].plot(L["t"], np.linalg.norm(L[f"{SIDE}_pdot"], axis=1), color=SLOT[i])
        axs[1].plot(L["t"], qd_ratio(L, meta), color=SLOT[i])
    axs[1].axhline(1.0, color=av.INK2, lw=1, ls=(0, (1, 2)))
    axs[1].text(data[runs[0][0]][0]["t"][-1], 1.0, " velocity limit", color=av.INK2, va="bottom", ha="right", fontsize=8)
    axs[0].set_ylabel(r"$|\dot{p}|$ [m/s]")
    axs[1].set_ylabel(r"$\max_i |\dot{q}_i| / \dot{q}_{\max,i}$")
    axs[0].set_title("With and without virtual spring saturation (left arm)")
    axs[1].set_xlabel("Time [s]")
    for ax in axs:
        av.phase_lines(ax, data[runs[0][0]][1])
    legend(axs[0], runs)
    fig.savefig(os.path.join(out, "cmp_spring.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    runs = [r for r in GROUPS["cmp_damping"] if r[0] in data]
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    a0, a1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    a2 = fig.add_subplot(gs[:, 1])
    a0.sharex(a1)
    err_axes([a0, a1], runs, data)
    a1.set_xlabel("Time [s]")
    a0.set_title(r"Joint damping design $C_0$ (left arm)")
    for i, (run, lbl) in enumerate(runs):
        L, _ = data[run]
        a2.plot(L[f"{SIDE}_p"][:, 0], L[f"{SIDE}_p"][:, 1], color=SLOT[i], lw=1.8)
    L, meta = data[runs[0][0]]
    for sg in meta["schedule"][1:]:
        i0 = np.searchsorted(L["t"], sg["t_start"] - 1e-6)
        a, b = L[f"{SIDE}_p"][i0], L[f"{SIDE}_pd"][i0]
        a2.plot([a[0], b[0]], [a[1], b[1]], color=av.MUTED, lw=1, ls=(0, (4, 3)))
        a2.plot(b[0], b[1], "x", color=av.INK, ms=7, mew=2)
    a2.set_aspect("equal", adjustable="datalim")
    a2.set_xlabel("x [m]")
    a2.set_ylabel("y [m]")
    a2.set_title("End-effector path (top view)")
    legend(a0, runs, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cmp_damping.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("comparison figures ->", out)


if __name__ == "__main__":
    main()
