# VSD control analysis — `reach__uniform_damping`

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
| Joint damping α | 1.0 → C₀ = diag(1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27) N·m·s/rad |
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
| reach | Left arm | 191.0 | 28.6 | 0.55 | 1.45 | 4.0 | 0.01 / 0.003 | 0.290 | 18.8 | 1.147 | 0.51 | 0.44 |
| reach | Right arm | 191.1 | 28.6 | 0.55 | 1.40 | 4.0 | 0.01 / 0.003 | 0.290 | 18.1 | 1.144 | 0.51 | 0.43 |
| return home | Left arm | 191.0 | 28.6 | 0.56 | 1.58 | 4.0 | 0.26 / 0.190 | 0.288 | 9.8 | 1.138 | 0.57 | 0.49 |
| return home | Right arm | 191.1 | 28.6 | 0.56 | 1.53 | 4.0 | 0.20 / 0.151 | 0.288 | 9.3 | 1.137 | 0.57 | 0.50 |

- Minimum distance between end-effectors: 0.332 m (repulsion threshold r_th = 0.3 m)
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
