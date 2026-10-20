# AERO-GUARD NEXUS — Step 66 Multi-Split Conformal RUL Evaluation

- Generated (UTC): `2026-10-19T22:02:37+00:00`
- Dataset: `NASA C-MAPSS FD001` (simulated benchmark; train file only)
- Repeats: **5**; target full-trajectory coverage: **80%**
- Splits meeting target: **5/5**
- Official test labels loaded: **No**
- Existing model files changed: **No**

## Per-split results

| Split | Seed | Engine coverage | Row coverage | Mean width (cycles) | Radius (cycles) | MAE | RMSE | Target met |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 6600 | 0.800 | 0.977 | 69.94 | 48.0136 | 12.338 | 18.086 | Yes |
| 2 | 6601 | 0.850 | 0.991 | 67.93 | 47.0543 | 11.034 | 15.870 | Yes |
| 3 | 6602 | 0.950 | 0.998 | 76.91 | 54.2234 | 11.303 | 16.121 | Yes |
| 4 | 6603 | 1.000 | 1.000 | 79.21 | 57.2058 | 10.970 | 15.546 | Yes |
| 5 | 6604 | 0.900 | 0.996 | 68.61 | 49.9516 | 10.898 | 16.043 | Yes |

## Mean across splits

| Metric | Mean | Population SD | Min | Max |
|---|---:|---:|---:|---:|
| conformal_radius_cycles | 51.2897 | 3.8489 | 47.0543 | 57.2058 |
| conformal_row_coverage | 0.9925 | 0.0081 | 0.9774 | 1.0 |
| conformal_full_trajectory_engine_coverage | 0.9 | 0.0707 | 0.8 | 1.0 |
| mean_conformal_interval_width_cycles | 72.5206 | 4.6289 | 67.9269 | 79.2145 |
| raw_member_p10_p90_row_coverage | 0.2873 | 0.0378 | 0.2383 | 0.3263 |
| raw_member_p10_p90_mean_width_cycles | 10.1218 | 1.3193 | 8.6378 | 12.4705 |
| point_mae_cycles | 11.3085 | 0.5325 | 10.8978 | 12.3376 |
| point_rmse_cycles | 16.3331 | 0.8984 | 15.5463 | 18.086 |

## Limitations

Repeated partitions reuse the same finite set of engines and are correlated. This is split-sensitivity evidence, not independent-trial confidence evidence.
Only the simulated C-MAPSS FD001 training trajectories were used. This does not validate real aircraft behavior, engineering limits, maintenance instructions or operational clearance.
