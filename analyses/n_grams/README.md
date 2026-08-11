# N-gram analysis

Pipeline for describing logs, extracting variant + n-gram (n1..n10) attachments,
and running Clauset-style power-law tests. Shared tools live under `cli/`.

```bash
# Full pipeline 01 -> 02 -> 03
python analyses/n_grams/main.py --datasets TEST_BPIC12

# Subset
python analyses/n_grams/main.py --datasets TEST_BPIC12 --from 02 --to 03
python analyses/n_grams/main.py --datasets TEST_BPIC12 --only 01
```

| Step | Script | Role |
|------|--------|------|
| 01 | `01_describe.py` | Event-log description via `cli/log_info.py` |
| 02 | `02_extract_ngrams.py` | Attachments for `variants` + `n1`..`n10` |
| 03 | `03_clauset_tests.py` | Clauset tests per concept via `cli/clauset_power_law.py` |

Later steps may evaluate at which n-gram length power laws appear.
