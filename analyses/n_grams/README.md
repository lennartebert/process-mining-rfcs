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

## Parallel mode (Slurm)

`--parallel` requires **exactly one** dataset. It writes per-log CSV shards (no LaTeX);
attachments and plots stay on their existing per-log paths.

| Writer | Shard | After combine |
|--------|-------|---------------|
| describe | `log_info_<LOG>.csv` | `log_info.csv` + `.tex` |
| Clauset | `<concept>/<model>/{gof,comparison,summary}_<LOG>.csv` | unsuffixed CSVs |
| compose | `variant_power_law_<LOG>.csv`, `log_n_fitted_types_<LOG>.csv`, `log_n_scaling_<LOG>.csv` | unsuffixed + `.tex` (scaling) |

```bash
# One log (local or Slurm array task)
python analyses/n_grams/main.py --datasets BPIC12 --parallel

# Smoke test in parallel mode
python analyses/n_grams/main.py --test --parallel

# After all shards exist, merge + write LaTeX
python analyses/n_grams/05_combine_parallel.py --output-dir results/n_grams
```

Slurm (from repo root):

```bash
mkdir -p .slurm/logs
sbatch .slurm/n_grams_parallel_array.sh   # one task per log
# ... wait until array finishes ...
sbatch .slurm/n_grams_combine.sh          # manual combine job
```
