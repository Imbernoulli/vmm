# Qwen3 MoE Router Margin Fragility

这个 probe 把 router top-k 的边界稳定性单独抽出来看：如果 top-1 margin 很小、Instruct/Coder route overlap 很低、router 权重位移又大，那么直接线性移动 router 很容易跨过离散 top-k 边界。这个分数是排序用的机制 proxy，不是最终下游分数。

## Result

- Status: `router_margin_fragility_rejects_direct_router_average`
- Router layers: `48`
- High-fragility layers: `24`
- Top fragile layer: `L17` score `0.7523`
- Least fragile layer: `L43` score `0.3886`
- Min safe-lambda proxy: `0.0197`
- Top category: `long_context` score `0.7329`

## Mechanism

Dense interpolation assumes one continuous parameter path can be evaluated by loss along the path. MoE router averaging adds a discrete top-k boundary: small logit changes can swap experts, and the wrong expert then receives the token. Therefore router movement needs a separate boundary/margin gate before it can be treated like ordinary Dense parameters.

## Layer Ranking

| layer | score | action | unsafe frac | mean margin | mean top-k Jaccard | mean top1 agreement | router rel | lambda proxy |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.7523 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0298 | 0.3949 | 0.3775 | 0.8444 | 0.0261 |
| 18 | 0.7267 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0402 | 0.3840 | 0.3528 | 0.8692 | 0.0391 |
| 12 | 0.7180 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0374 | 0.4167 | 0.3456 | 0.8235 | 0.0338 |
| 20 | 0.7157 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0397 | 0.4015 | 0.3222 | 0.8089 | 0.0439 |
| 11 | 0.7151 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0372 | 0.4259 | 0.3456 | 0.8191 | 0.0352 |
| 13 | 0.7122 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0398 | 0.4240 | 0.3553 | 0.8553 | 0.0384 |
| 15 | 0.7040 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0426 | 0.4060 | 0.3342 | 0.8246 | 0.0438 |
| 4 | 0.7026 | `freeze_router` | 0.9167 | 0.0307 | 0.4725 | 0.3832 | 0.7927 | 0.0308 |
| 19 | 0.6911 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0466 | 0.3892 | 0.3422 | 0.8252 | 0.0436 |
| 29 | 0.6910 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0289 | 0.4360 | 0.3032 | 0.5621 | 0.0396 |
| 25 | 0.6884 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0399 | 0.4267 | 0.2904 | 0.7017 | 0.0459 |
| 24 | 0.6790 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0503 | 0.3559 | 0.2938 | 0.7478 | 0.0392 |
| 22 | 0.6760 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0624 | 0.2840 | 0.2473 | 0.7937 | 0.0500 |
| 14 | 0.6738 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0521 | 0.4015 | 0.3263 | 0.8413 | 0.0444 |
| 47 | 0.6664 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0275 | 0.4854 | 0.3936 | 0.5914 | 0.0324 |
| 16 | 0.6524 | `freeze_router_prioritize_calibration` | 1.0000 | 0.0508 | 0.4100 | 0.3991 | 0.8176 | 0.0500 |

## Category Ranking

| category | score | unsafe frac | mean margin | mean top-k Jaccard | mean top1 agreement | calibrate rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `long_context` | 0.7329 | 0.9167 | 0.0421 | 0.4496 | 0.3992 | 44 |
| `safety` | 0.6816 | 0.8750 | 0.0436 | 0.4485 | 0.3931 | 83 |
| `agentic_code` | 0.6729 | 0.9583 | 0.0447 | 0.4278 | 0.3646 | 46 |
| `legal` | 0.6547 | 0.8125 | 0.0437 | 0.4554 | 0.4281 | 39 |
| `code` | 0.6043 | 0.8854 | 0.0460 | 0.4384 | 0.4135 | 84 |
| `finance` | 0.5770 | 0.7708 | 0.0455 | 0.4849 | 0.4492 | 36 |
| `general` | 0.4771 | 0.8854 | 0.0500 | 0.4459 | 0.4067 | 85 |
| `math` | 0.4691 | 0.8229 | 0.0492 | 0.4819 | 0.4414 | 76 |

## Most Fragile Observed Slices

| category | prompt | layer | score | margin | top-k Jaccard | top1 agreement | action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `agentic_code` | 10 | 14 | 0.8147 | 0.0422 | 0.3477 | 0.0769 | `calibrate_router_before_average` |
| `code` | 4 | 20 | 0.8073 | 0.0375 | 0.3855 | 0.0690 | `calibrate_router_before_average` |
| `safety` | 6 | 22 | 0.8064 | 0.0412 | 0.2581 | 0.1786 | `calibrate_router_before_average` |
| `math` | 3 | 18 | 0.8051 | 0.0382 | 0.3438 | 0.1818 | `calibrate_router_before_average` |
| `long_context` | 11 | 19 | 0.8014 | 0.0359 | 0.3793 | 0.1290 | `calibrate_router_before_average` |
| `code` | 4 | 15 | 0.7993 | 0.0412 | 0.3415 | 0.1379 | `calibrate_router_before_average` |
| `code` | 5 | 18 | 0.7971 | 0.0356 | 0.3738 | 0.2000 | `calibrate_router_before_average` |
| `agentic_code` | 10 | 17 | 0.7957 | 0.0250 | 0.3113 | 0.3462 | `calibrate_router_before_average` |
| `safety` | 6 | 17 | 0.7943 | 0.0262 | 0.3876 | 0.2500 | `calibrate_router_before_average` |
| `general` | 1 | 17 | 0.7928 | 0.0284 | 0.3520 | 0.2800 | `calibrate_router_before_average` |
| `safety` | 6 | 15 | 0.7907 | 0.0374 | 0.3926 | 0.1429 | `calibrate_router_before_average` |
| `code` | 5 | 14 | 0.7870 | 0.0374 | 0.3717 | 0.2000 | `calibrate_router_before_average` |

## Interpretation

The most fragile layer is `L17`; the least fragile layer is `L43`, but every layer still has at least one unsafe observed category/prompt slice. The algorithmic consequence is unchanged but sharper: direct router averaging remains rejected, and any router movement must be a calibrated route-KD/HARC-style intervention with the same eval-bundle/source-dominance gate as other candidates.

## Outputs

- `layer_fragility`: `results/qwen3_moe_router_margin_fragility/layer_margin_fragility.csv`
- `category_fragility`: `results/qwen3_moe_router_margin_fragility/category_margin_fragility.csv`
- `slice_fragility`: `results/qwen3_moe_router_margin_fragility/slice_margin_fragility.csv`
- `literature_sources`: `results/qwen3_moe_router_margin_fragility/literature_sources.json`
- `figure`: `results/qwen3_moe_router_margin_fragility/router_margin_fragility.png`
- `summary`: `results/qwen3_moe_router_margin_fragility/summary.json`
- `report`: `results/qwen3_moe_router_margin_fragility/report.md`
