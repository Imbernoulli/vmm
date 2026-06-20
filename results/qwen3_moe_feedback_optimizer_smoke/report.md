# Qwen3 MoE Feedback Optimizer Smoke

- Status: `passed`
- Cases passed: `10/10`

| case | assertion | expected | actual | passed |
| --- | --- | --- | --- | --- |
| `code_regression_restore` | `code_scale_increases` | `>0.50` | `0.589298` | `True` |
| `math_regression_shrink` | `math_scale_decreases` | `<0.80` | `0.748692` | `True` |
| `hard_cap` | `max_delta_capped` | `<=0.65` | `0.471438` | `True` |
| `awaiting_eval` | `no_feedback_without_scores` | `0 changed groups` | `0` | `True` |
| `integration_eval_bundle` | `humaneval_regression_detected` | `source_frontier_regression` | `source_frontier_regression` | `True` |
| `integration_eval_bundle` | `safety_regression_detected` | `source_frontier_regression` | `source_frontier_regression` | `True` |
| `integration_eval_bundle` | `paired_predictions_loaded` | `negative humaneval paired delta` | `-0.5000` | `True` |
| `integration_eval_bundle` | `code_scale_restored` | `>0.50` | `0.589560` | `True` |
| `integration_eval_bundle` | `safety_scale_shrunk` | `<1.00` | `0.963964` | `True` |
| `integration_eval_bundle` | `materialization_gate_opens` | `materialize_feedback_candidate` | `materialize_feedback_candidate` | `True` |
