import argparse
import re

import mujoco
import numpy as np
from scipy.optimize import least_squares

from common import SCENE_XML, ARM_SIDES, arm_dofs, finger_joint_key, site_name

EE_POS = np.array([0.38, 0.27, 0.98])
ELBOW_POS = np.array([0.02, 0.33, 0.93])
R_HOME = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float)

FINGER_HOME = {"j0": 0.0, "j1": 0.3, "j2": 0.3, "j3": 0.3,
               "thj0": 0.7, "thj1": 0.3, "thj2": 0.3, "thj3": 0.3}


def mirror(v):
    return v * np.array([1, -1, 1])


def solve_arm(m, side, seeds=40, rng=np.random.default_rng(0)):
    d = mujoco.MjData(m)
    dof = arm_dofs(m, side)
    qadr = np.array([m.jnt_qposadr[m.dof_jntid[i]] for i in dof])
    sid = m.site(site_name(side)).id
    elbow = m.body(f"forearm_link_{side[0]}").id
    sgn = 1 if side == "left" else -1
    p_ee = EE_POS if side == "left" else mirror(EE_POS)
    p_el = ELBOW_POS if side == "left" else mirror(ELBOW_POS)

    lo = np.full(7, -2 * np.pi); hi = np.full(7, 2 * np.pi)
    for k, i in enumerate(dof):
        j = m.dof_jntid[i]
        if m.jnt_limited[j]:
            lo[k], hi[k] = m.jnt_range[j] * 0.95

    def residual(q):
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[sid].reshape(3, 3)
        return np.concatenate([
            3.0 * (d.site_xpos[sid] - p_ee),
            1.0 * (d.xpos[elbow] - p_el),
            0.5 * (R - R_HOME).ravel(),
            0.01 * q,
        ])

    best = None
    for _ in range(seeds):
        q0 = rng.uniform(lo, hi) * 0.5
        sol = least_squares(residual, q0, bounds=(lo, hi))
        if best is None or sol.cost < best.cost:
            best = sol
    q = (best.x + np.pi) % (2 * np.pi) - np.pi
    q = np.clip(q, lo, hi)
    d.qpos[qadr] = q
    mujoco.mj_kinematics(m, d)
    print(f"{side}: q = {np.round(q, 4)}  ee = {np.round(d.site_xpos[sid], 3)} "
          f"(goal {np.round(p_ee, 3)}), cost = {best.cost:.2e}")
    return qadr, q, sgn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    qpos = m.qpos0.copy()
    for side in ARM_SIDES:
        qadr, q, _ = solve_arm(m, side)
        qpos[qadr] = q
    for j in range(m.njnt):
        name = m.joint(j).name
        if name.startswith(("left_", "right_")) and "_joint_" not in name:
            qpos[m.jnt_qposadr[j]] = np.clip(FINGER_HOME[finger_joint_key(name)], *m.jnt_range[j])

    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    print("contacts at home:", d.ncon)

    key = ('  <keyframe>\n    <key name="home" qpos="' +
           " ".join(f"{v:.6g}" for v in qpos) + '"/>\n  </keyframe>\n')
    if args.dry_run:
        print(key)
        return
    s = open(SCENE_XML).read()
    s = re.sub(r"  <keyframe>.*?</keyframe>\n", "", s, flags=re.S)
    s = s.replace("</mujoco>", key + "</mujoco>")
    open(SCENE_XML, "w").write(s)
    print("wrote keyframe 'home' to", SCENE_XML)


if __name__ == "__main__":
    main()
