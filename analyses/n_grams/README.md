# N-gram analysis

Pipeline for describing logs, extracting variant + n-gram (n1..n10) attachments,
running Clauset-style power-law tests, and composing summary tables/plots.
Shared tools live under `cli/`.

```bash
# Full pipeline 01 -> 02 -> 03 -> 04
python analyses/n_grams/main.py --datasets TEST_BPIC12

# Restrict concepts (default: n1..n10 + variants)
python analyses/n_grams/main.py --datasets TEST_BPIC12 --concepts n1 n2 variants

# Quick smoke test (TEST_BPIC12; concepts n1, n2, variants; fewer bootstraps)
python analyses/n_grams/main.py --test

# Subset
python analyses/n_grams/main.py --datasets TEST_BPIC12 --from 02 --to 04
python analyses/n_grams/main.py --datasets TEST_BPIC12 --only 04
```

| Step | Script | Role |
|------|--------|------|
| 01 | `01_describe.py` | Event-log description via `cli/log_info.py` → `results/n_grams/log_info.*` |
| 02 | `02_extract_ngrams.py` | Attachments under `results/attachments/<concept>/<log>/` |
| 03 | `03_clauset_tests.py` | Clauset fits under `results/n_grams/<concept>/<model>/` |
| 04 | `04_compose_results.py` | Tables + plots under `results/n_grams/` |

Step 03 model policy: **variants** fit all three (`full_range`, `lower_bounded`, `doubly_bounded`); other n-grams fit **only** `lower_bounded_power_law`.

Step 04 defaults to `lower_bounded_power_law` (`--power-law-model` to change).

`--test` writes attachments to `results/test/attachments/` and describe/Clauset/compose
to `results/test/n_grams/` (including `log_info.csv` / `.tex`).
