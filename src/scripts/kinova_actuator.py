import mujoco
import numpy as np


class KinovaTorqueActuators:
    def __init__(self, m, dofs, acts, mode="torque", ki=150.0, leak_tc=1.0):
        self.m = m
        self.dofs = np.asarray(dofs)
        self.acts = np.asarray(acts)
        self.mode = mode
        self.ki = ki
        self.lam = np.exp(-m.opt.timestep / leak_tc)
        self.gear = m.actuator_gear[self.acts, 0].copy()
        self.lo = m.actuator_ctrlrange[self.acts, 0] * self.gear
        self.hi = m.actuator_ctrlrange[self.acts, 1] * self.gear
        self.armature = m.dof_armature[self.dofs].copy()
        self.damping = m.dof_damping[self.dofs].copy()
        self.integ = np.zeros(len(self.dofs))
        self.tau_m = np.zeros(len(self.dofs))
        self._row_of_dof = {}

    def sensor_torque(self, d):
        f_loss = np.zeros(len(self.dofs))
        pos = {dof: k for k, dof in enumerate(self.dofs)}
        for r in range(d.nefc):
            if d.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF and d.efc_id[r] in pos:
                f_loss[pos[d.efc_id[r]]] = -d.efc_force[r]
        tau_act = d.actuator_force[self.acts] * self.gear
        return (tau_act - self.damping * d.qvel[self.dofs] - f_loss
                - self.armature * d.qacc[self.dofs])

    def command(self, d, tau_d):
        tau_d = np.clip(tau_d, self.lo, self.hi)
        tau_s = self.sensor_torque(d)
        if self.mode == "torque":
            self.integ = self.lam * self.integ + self.ki * self.m.opt.timestep * (tau_d - tau_s)
            self.integ = np.clip(self.integ, self.lo, self.hi)
            tau_m = tau_d + self.integ
        else:
            tau_m = tau_d
        self.tau_m = np.clip(tau_m, self.lo, self.hi)
        return self.tau_m / self.gear, self.tau_m, tau_s
