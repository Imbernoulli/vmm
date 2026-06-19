# Qwen3 MoE Tail-Trimmed Expert-Only Candidate

This candidate starts from the materialized expert-only trust-region candidate and trims only the remaining high routed-expert delta tail.
Shared attention remains frozen, router remains frozen, and the output checkpoint keeps the same model structure.

- Status: `tail_trimmed_rules_ready`
- Target routed tensor cap: `0.650`
- Expert groups: `6144`
- Scaled expert groups: `140`
- Scaled rules: `140`
- Source total relative delta norm: `0.246`
- Estimated total relative delta norm: `0.243`
- Estimated routed max relative delta: `0.650`
- Estimated routed tensors >0.65: `0`
- Writer dry-run: `True`
- Materialized: `True`

## Top Scaled Expert Groups

| layer | expert | tensors | max rel | scale | predicted max rel |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 56 | 3 | 0.750 | 0.867 | 0.650 |
| 6 | 112 | 3 | 0.750 | 0.867 | 0.650 |
| 42 | 87 | 3 | 0.750 | 0.867 | 0.650 |
| 8 | 15 | 3 | 0.750 | 0.867 | 0.650 |
| 32 | 15 | 3 | 0.750 | 0.867 | 0.650 |
| 32 | 0 | 3 | 0.750 | 0.867 | 0.650 |
| 45 | 40 | 3 | 0.750 | 0.867 | 0.650 |
| 33 | 109 | 3 | 0.750 | 0.867 | 0.650 |
| 47 | 65 | 3 | 0.750 | 0.867 | 0.650 |
| 9 | 99 | 3 | 0.750 | 0.867 | 0.650 |
| 6 | 120 | 3 | 0.750 | 0.867 | 0.650 |
| 42 | 88 | 3 | 0.750 | 0.867 | 0.650 |
| 2 | 101 | 3 | 0.750 | 0.867 | 0.650 |
| 37 | 51 | 3 | 0.750 | 0.867 | 0.650 |
| 2 | 33 | 3 | 0.750 | 0.867 | 0.650 |
| 6 | 111 | 3 | 0.750 | 0.867 | 0.650 |
| 2 | 83 | 3 | 0.750 | 0.867 | 0.650 |
| 8 | 44 | 3 | 0.750 | 0.867 | 0.650 |
| 8 | 118 | 3 | 0.750 | 0.867 | 0.650 |
| 39 | 119 | 3 | 0.750 | 0.867 | 0.650 |
| 44 | 117 | 3 | 0.750 | 0.867 | 0.650 |
| 5 | 89 | 3 | 0.750 | 0.867 | 0.650 |
| 43 | 23 | 3 | 0.750 | 0.867 | 0.650 |
| 0 | 20 | 3 | 0.750 | 0.867 | 0.650 |
| 8 | 119 | 3 | 0.750 | 0.867 | 0.650 |

## Files

- `expert_tail_scales`: `results/qwen3_moe_tail_trimmed_expert_only_candidate/expert_tail_scales.csv`
- `tail_trimmed_rules`: `results/qwen3_moe_tail_trimmed_expert_only_candidate/tail_trimmed_rules.csv`
- `tensor_rules`: `results/qwen3_moe_tail_trimmed_expert_only_candidate/tensor_rules.txt`
- `writer_command`: `results/qwen3_moe_tail_trimmed_expert_only_candidate/writer_command.txt`
- `dry_run_command`: `results/qwen3_moe_tail_trimmed_expert_only_candidate/dry_run_command.txt`
- `dry_run_manifest`: `results/qwen3_moe_tail_trimmed_expert_only_candidate/dry_run/merge_manifest.json`
- `materialized_manifest`: `results/checkpoints/qwen3_moe_tail_trimmed_expert_only_candidate/merge_manifest.json`
- `summary`: `results/qwen3_moe_tail_trimmed_expert_only_candidate/summary.json`
- `report`: `results/qwen3_moe_tail_trimmed_expert_only_candidate/report.md`
