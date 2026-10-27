# VSD control analysis — `repulsion`

## Controller
VSD control law (Arimoto & Sekimoto):

```
τ = −C₀q̇ − Jᵀ(k·Δp + ζ√k·ṗ) + Jωᵀ(k_o·e_o − ζ_o√k_o·ω) + g(q) + τ_fric + Jᵀf_rep
C₀ = α·√H₀ ,  H₀ = diag(Σⱼ H_M,ij)
```

| Parameter | Value |
|---|---|
| Task-space spring k | 600.0 N/m (saturation 15.0 N) |
| Task-space damping ζ | 1.5 (c = ζ√k = 36.7 N·s/m) |
| Joint damping α | 1.0 → C₀ = diag(2.07, 2.06, 1.37, 1.54, 0.79, 0.86, 0.22) N·m·s/rad |
| Orientation spring k_o / ζ_o | 30.0 N·m/rad (saturation 3.0 N·m) / 1.0 |
| Repulsion k_rep / r_th | 3.75 N·m / 0.3 m |
| Actuator mode | torque (friction compensation off) |

## Actuator model (Kinova Gen3)
| Item | Large (1–4) | Small (5–7) |
|---|---|---|
| Gear ratio | 100:1 | 100:1 |
| Torque limit | 39 Nm | 9 Nm |
| Velocity limit | 1.39 rad/s | 1.22 rad/s |
| Reflected rotor inertia | 0.015 kg·m² | 0.01928 kg·m² |
| Friction (Coulomb / viscous) | 2.15 Nm / 2.0 Nm·s/rad | 1.6 Nm / 1.36 Nm·s/rad |

## Step-response metrics

| Segment | Arm | Step [mm] | Orientation step [deg] | Rise time 10–90% [s] | Settling time 2% [s] | Overshoot [%] | Steady-state error [mm] / [deg] | Peak EE speed [m/s] | Max path deviation [mm] | Path length / straight line | Peak q̇/limit | Peak τ/limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hold home | Left arm | 0.0 | – | – | – | – | – | – | – | – | – | – |
| hold home | Right arm | 0.0 | – | – | – | – | – | – | – | – | – | – |
| crossing targets | Left arm | 336.5 | 0.0 | – | – | 0.0 | 174.07 / 0.115 | 0.278 | 57.7 | 0.627 | 0.39 | 0.43 |
| crossing targets | Right arm | 336.7 | 0.0 | – | – | 0.0 | 177.33 / 0.068 | 0.273 | 58.4 | 0.614 | 0.40 | 0.42 |
| return home | Left arm | 181.7 | 0.1 | 0.53 | 0.96 | 2.8 | 0.01 / 0.004 | 0.383 | 17.9 | 1.113 | 0.39 | 0.57 |
| return home | Right arm | 179.1 | 0.1 | 0.53 | 0.95 | 2.8 | 0.02 / 0.005 | 0.378 | 16.7 | 1.111 | 0.39 | 0.56 |

- Minimum distance between end-effectors: 0.251 m (repulsion threshold r_th = 0.3 m)
- Maximum number of contacts: 0

## Figures

![01_ee_position.png](01_ee_position.png)
![02_error_norm.png](02_error_norm.png)
![03_ee_speed_accel.png](03_ee_speed_accel.png)
![04_joint_angle.png](04_joint_angle.png)
![05_joint_velocity.png](05_joint_velocity.png)
![06_joint_torque.png](06_joint_torque.png)
![07_phase_task.png](07_phase_task.png)
![08_phase_joint.png](08_phase_joint.png)
![09_ee_path.png](09_ee_path.png)
![10_dual_arm_distance.png](10_dual_arm_distance.png)
