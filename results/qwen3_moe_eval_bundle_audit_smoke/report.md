# Qwen3 MoE vLLM Eval Bundle Audit Smoke

- Status: `passed`
- Cases: `6/6`

| case | status | usable | expected | passed |
| --- | --- | ---: | ---: | --- |
| `valid_bundle` | `passed` | 3 | 3 | `True` |
| `stale_model` | `invalid_bundle` | 2 | 2 | `True` |
| `missing_task` | `invalid_bundle` | 2 | 2 | `True` |
| `low_examples` | `invalid_bundle` | 2 | 2 | `True` |
| `key_mismatch` | `invalid_bundle` | 2 | 2 | `True` |
| `manifest_mismatch` | `invalid_bundle` | 2 | 2 | `True` |
