# Qwen3 MoE Router Move Gate

这个 gate 专门回答一个问题：在已经有 routed-expert candidate 的情况下，能不能把 router 也作为 same-shape average 的一部分打开。

- Status: `router_move_rejected_freeze_router`
- Router layers: `48`
- Allowed router layers: `0`
- Frozen router layers: `48`
- Total router relative delta norm: `0.7393`
- Mean top-k Jaccard: `0.4539`
- Min top-k Jaccard: `0.2422`
- Mean top1 agreement: `0.4125`
- Min top1 agreement: `0.0690`

## Decision

No router layer passes the all-observed-category guard. Because router tensors are shared across categories, selective category-level movement is not expressible as a same-shape weight average. The current unified Qwen3 MoE rule should keep router frozen and test route-KD/HARC-style calibration only as a separate trained/calibrated intervention.

## Layer Gate

| layer | decision | lambda | unsafe | calibrate | freeze | small | passed | mean Jaccard | min top1 | router rel | reason |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | `freeze_router` | 0.0000 | 3 | 3 | 0 | 9 | 0 | 0.6088 | 0.3846 | 0.8763 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 41 | `freeze_router` | 0.0000 | 5 | 5 | 0 | 7 | 0 | 0.5850 | 0.3939 | 0.2988 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 43 | `freeze_router` | 0.0000 | 5 | 5 | 0 | 3 | 4 | 0.5478 | 0.3793 | 0.2796 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 42 | `freeze_router` | 0.0000 | 6 | 6 | 0 | 2 | 4 | 0.5547 | 0.2727 | 0.2906 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 3 | `freeze_router` | 0.0000 | 6 | 6 | 0 | 1 | 5 | 0.5250 | 0.3600 | 0.7497 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 2 | `freeze_router` | 0.0000 | 7 | 7 | 0 | 0 | 5 | 0.5044 | 0.3571 | 0.7799 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 44 | `freeze_router` | 0.0000 | 7 | 7 | 0 | 4 | 1 | 0.5167 | 0.3438 | 0.2837 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 1 | `freeze_router` | 0.0000 | 7 | 7 | 0 | 5 | 0 | 0.5043 | 0.3846 | 0.7390 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 40 | `freeze_router` | 0.0000 | 8 | 8 | 0 | 1 | 3 | 0.5305 | 0.3793 | 0.3128 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 37 | `freeze_router` | 0.0000 | 9 | 7 | 2 | 0 | 3 | 0.5535 | 0.4318 | 0.3444 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 5 | `freeze_router` | 0.0000 | 9 | 8 | 1 | 3 | 0 | 0.5015 | 0.4194 | 0.7841 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 32 | `freeze_router` | 0.0000 | 9 | 9 | 0 | 2 | 1 | 0.4622 | 0.4000 | 0.4059 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 8 | `freeze_router` | 0.0000 | 10 | 10 | 0 | 2 | 0 | 0.4954 | 0.4000 | 0.7833 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 7 | `freeze_router` | 0.0000 | 10 | 10 | 0 | 1 | 1 | 0.4717 | 0.2857 | 0.7842 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 46 | `freeze_router` | 0.0000 | 10 | 10 | 0 | 2 | 0 | 0.4680 | 0.2400 | 0.3303 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 39 | `freeze_router` | 0.0000 | 11 | 11 | 0 | 1 | 0 | 0.5353 | 0.2692 | 0.3109 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 38 | `freeze_router` | 0.0000 | 11 | 10 | 1 | 0 | 1 | 0.5077 | 0.2308 | 0.3289 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 45 | `freeze_router` | 0.0000 | 11 | 11 | 0 | 1 | 0 | 0.4913 | 0.2903 | 0.2947 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 6 | `freeze_router` | 0.0000 | 11 | 10 | 1 | 0 | 1 | 0.4730 | 0.3462 | 0.8064 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 4 | `freeze_router` | 0.0000 | 11 | 11 | 0 | 1 | 0 | 0.4725 | 0.2188 | 0.7927 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 9 | `freeze_router` | 0.0000 | 11 | 11 | 0 | 1 | 0 | 0.4609 | 0.2500 | 0.7872 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 31 | `freeze_router` | 0.0000 | 11 | 11 | 0 | 0 | 1 | 0.4267 | 0.1923 | 0.4521 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 34 | `freeze_router` | 0.0000 | 11 | 11 | 0 | 0 | 1 | 0.4265 | 0.2727 | 0.3659 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 33 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4792 | 0.1154 | 0.3730 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 47 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4854 | 0.2308 | 0.5914 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 35 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4776 | 0.2000 | 0.3538 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 10 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4485 | 0.3333 | 0.8217 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 29 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4360 | 0.1923 | 0.5621 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 36 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4328 | 0.3600 | 0.3488 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 25 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4267 | 0.2000 | 0.7017 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 11 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4259 | 0.2759 | 0.8191 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 13 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4240 | 0.1935 | 0.8553 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 12 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4167 | 0.2258 | 0.8235 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 16 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4100 | 0.2800 | 0.8176 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 15 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4060 | 0.1379 | 0.8246 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 30 | `freeze_router` | 0.0000 | 12 | 11 | 1 | 0 | 0 | 0.4036 | 0.1818 | 0.5164 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 14 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4015 | 0.0769 | 0.8413 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 20 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4015 | 0.0690 | 0.8089 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 27 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4013 | 0.1923 | 0.6012 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 28 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.4011 | 0.1600 | 0.5648 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 26 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.3994 | 0.2308 | 0.6374 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 17 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.3949 | 0.2500 | 0.8444 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 19 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.3892 | 0.1290 | 0.8252 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 18 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.3840 | 0.1818 | 0.8692 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 21 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.3722 | 0.2273 | 0.7861 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 24 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.3559 | 0.1923 | 0.7478 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 23 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.3072 | 0.2308 | 0.7700 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |
| 22 | `freeze_router` | 0.0000 | 12 | 12 | 0 | 0 | 0 | 0.2840 | 0.1333 | 0.7937 | At least one observed category requires calibration/freeze; the router tensor is shared across categories. |

## Router Delta Summary

| layer | router rel | max abs delta | numel |
| ---: | ---: | ---: | ---: |
| 0 | 0.8763 | 1.1006 | 262144 |
| 18 | 0.8692 | 1.3151 | 262144 |
| 13 | 0.8553 | 1.2197 | 262144 |
| 17 | 0.8444 | 1.2842 | 262144 |
| 14 | 0.8413 | 1.2515 | 262144 |
| 19 | 0.8252 | 1.2422 | 262144 |
| 15 | 0.8246 | 1.2190 | 262144 |
| 12 | 0.8235 | 1.2373 | 262144 |
| 10 | 0.8217 | 1.5073 | 262144 |
| 11 | 0.8191 | 1.4827 | 262144 |
| 16 | 0.8176 | 1.2070 | 262144 |
| 20 | 0.8089 | 1.3213 | 262144 |

## Mechanism

Router tensor 是整层共享的参数；如果同一层里任何 category/prompt slice 已经触发 `calibrate_router_before_average` 或 `freeze_router_and_check_load_balance`，就不能只对安全 category 移动这层 router。当前真实 Qwen3 probe 中没有任何一层在全部观察场景里通过，因此统一规则仍应冻结 router。下一步如果要打开 router，应该做 route-KD / HARC-style calibration 并重新 probe，而不是直接平均 router weights。

## Outputs

- `report`: `results/qwen3_moe_router_move_gate/report.md`
- `summary`: `results/qwen3_moe_router_move_gate/summary.json`
- `router_layer_move_gate`: `results/qwen3_moe_router_move_gate/router_layer_move_gate.csv`
- `router_delta_summary`: `results/qwen3_moe_router_move_gate/router_delta_summary.csv`
