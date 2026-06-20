# Qwen3 MoE HARC Readiness Smoke

- Status: `smoke_passed`
- Cases: `4/4`

| case | expected | actual | passed |
| --- | --- | --- | --- |
| `harc_ready_waiting_cache` | `harc_ready_for_curvature_collection_waiting_cache` | `harc_ready_for_curvature_collection_waiting_cache` | `True` |
| `harc_solver_ready` | `harc_ready_for_matrix_free_solver` | `harc_ready_for_matrix_free_solver` | `True` |
| `router_safe` | `harc_not_recommended_router_average_not_rejected` | `harc_not_recommended_router_average_not_rejected` | `True` |
| `no_repair` | `harc_not_recommended_no_local_repair_signal` | `harc_not_recommended_no_local_repair_signal` | `True` |
