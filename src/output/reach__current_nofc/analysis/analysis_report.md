# VSD control analysis — `reach__current_nofc`

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
| Actuator mode | current (friction compensation off) |

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
| reach | Left arm | 191.0 | 28.6 | – | – | 0.0 | 43.75 / 20.984 | 0.092 | 37.7 | 0.890 | 0.15 | 0.38 |
| reach | Right arm | 191.1 | 28.6 | – | – | 0.0 | 44.96 / 20.516 | 0.089 | 37.7 | 0.885 | 0.15 | 0.38 |
| return home | Left arm | 154.1 | 10.5 | – | – | 0.0 | 18.47 / 2.913 | 0.080 | 31.0 | 0.998 | 0.14 | 0.38 |
| return home | Right arm | 153.2 | 10.9 | – | – | 0.0 | 18.23 / 3.030 | 0.081 | 31.5 | 1.002 | 0.14 | 0.37 |

- Minimum distance between end-effectors: 0.377 m (repulsion threshold r_th = 0.3 m)
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
