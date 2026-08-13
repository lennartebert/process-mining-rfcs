# N-gram analysis

Pipeline for describing logs, extracting variant + n-gram (n1..n10) attachments,
running Clauset-style power-law tests, combining parallel shards, and composing
summary tables/plots. Shared tools live under `cli/`.

```bash
# Full serial pipeline 01 -> 02 -> 03 -> 05 (skips combine)
python analyses/n_grams/main.py --datasets TEST_BPIC12

# Restrict concepts (default: n1..n10 + variants)
python analyses/n_grams/main.py --datasets TEST_BPIC12 --concepts n1 n2 variants

# Quick smoke test (TEST_BPIC12; concepts n1, n2, variants; fewer bootstraps)
python analyses/n_grams/main.py --test

# Subset
python analyses/n_grams/main.py --datasets TEST_BPIC12 --from 02 --to 03
python analyses/n_grams/main.py --datasets TEST_BPIC12 --only 05
```

| Step | Script | Role |
|------|--------|------|
| 01 | `01_describe.py` | Event-log description via `cli/log_info.py` → `results/n_grams/log_info.*` |
| 02 | `02_extract_ngrams.py` | Attachments under `results/attachments/<concept>/<log>/` |
| 03 | `03_clauset_tests.py` | Clauset fits under `results/n_grams/<concept>/<model>/` |
| 04 | `04_combine_parallel.py` | Merge parallel `log_info` + Clauset shards (after `--parallel`) |
| 05 | `05_compose_results.py` | Tables + plots under `results/n_grams/` |

Step 03 model policy: **variants** fit all three (`full_range`, `lower_bounded`, `doubly_bounded`); other n-grams fit **only** `lower_bounded_power_law`.

Step 05 defaults to `lower_bounded_power_law` (`--power-law-model` to change).

`--test` writes attachments to `results/test/attachments/` and describe/Clauset/compose
to `results/test/n_grams/` (including `log_info.csv` / `.tex`). That smoke path uses
the real `TEST_BPIC12` log. It is not the synthetic Zipf E2E.

Synthetic full-range PL data (known Zipf type frequencies, log name `TEST`) is a
unittest, not a `main.py` flag:

```bash
python -m unittest tests.test_pl_synthetic
# slow Monte Carlo calibration (not default CI)
RUN_PL_CALIBRATION=1 python -m unittest tests.test_pl_synthetic.SyntheticPLCalibrationTests
python tests/test_pl_synthetic.py --calibrate
```

Attachments land under `results/test/attachments/<concept>/TEST/`; Clauset CSVs
under `results/test/n_grams/`.

## Parallel mode (Slurm)

`--parallel` requires **exactly one** dataset and runs **01 → 02 → 03** only
(per-log CSV shards, no LaTeX). After all array tasks finish, run `--combine`
to execute **04 → 05** (merge shards, then compose tables/plots).

| Writer | Shard | After `--combine` (step 04) |
|--------|-------|-----------------------------|
| describe | `log_info_<LOG>.csv` | `log_info.csv` + `.tex` |
| Clauset | `<concept>/<model>/{gof,comparison,summary}_<LOG>.csv` | unsuffixed CSVs |

Step 05 then builds `variant_power_law.*`, `log_n_scaling.*`, `pl_outcome_summary.*`, plots, etc. from the combined summaries.

```bash
# One log (local or Slurm array task)
python analyses/n_grams/main.py --datasets BPIC12 --parallel

# Smoke test in parallel mode
python analyses/n_grams/main.py --test --parallel

# After all shards exist: merge + compose
python analyses/n_grams/main.py --combine --output-dir results/n_grams
```

Slurm (from repo root):

```bash
mkdir -p .slurm/logs
sbatch .slurm/n_grams_parallel_array.sh   # one task per log (01-03)
# ... wait until array finishes ...
sbatch .slurm/n_grams_combine.sh          # --combine: 04 then 05
```
