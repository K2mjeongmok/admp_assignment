import os

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(SRC_DIR, "xml", "scene.xml")
OUTPUT_DIR = os.path.join(SRC_DIR, "output")

ARM_SIDES = ("left", "right")


def site_name(side):
    return f"ee_site_{side[0]}"


def arm_joint_names(side):
    return [f"{side}_joint_{i}" for i in range(1, 8)]


def arm_dofs(m, side):
    return np.array([m.jnt_dofadr[m.joint(n).id] for n in arm_joint_names(side)])


def arm_actuators(m, side):
    return np.array([m.actuator(f"{side}_joint_{i}_motor").id for i in range(1, 8)])


def finger_joint_key(joint_name):
    j = joint_name.split("_", 1)[1]
    return j if j.startswith("th") else j[-2:]
