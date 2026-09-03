# N-gram analysis

Pipeline for describing logs, extracting variant + n-gram (n1..n10) attachments,
running Clauset-style power-law tests, combining parallel shards, and composing
summary tables/plots. Shared tools live under `cli/`.

Describe / Clauset / compose outputs go under `results/n_grams/real` for real
logs and `results/n_grams/sim` for `*_sim` logs. Attachments stay under
`results/attachments/<concept>/<log>/`.

```bash
# Full serial pipeline 01 -> 02 -> 03 -> 05 (skips combine)
python analyses/n_grams/main.py --datasets TEST_BPIC12

# All real benchmark logs (writes results/n_grams/real)
python analyses/n_grams/main.py --datasets ALL_REAL_LOGS

# First-order DFG-simulated counterparts (writes results/n_grams/sim)
python analyses/n_grams/06_generate_synthetic.py
python analyses/n_grams/main.py --datasets ALL_SIM_LOGS

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
| 01 | `01_describe.py` | Event-log description via `cli/log_info.py` → `results/n_grams/real/log_info.*` |
| 02 | `02_extract_ngrams.py` | Attachments under `results/attachments/<concept>/<log>/` |
| 03 | `03_clauset_tests.py` | Clauset fits under `results/n_grams/real/<concept>/<model>/` |
| 04 | `04_combine_parallel.py` | Merge parallel `log_info` + Clauset shards (after `--parallel`) |
| 05 | `05_compose_results.py` | Tables + plots under `results/n_grams/real/` |
| 06 | `06_generate_synthetic.py` | Standalone DFG simulator → `data/synthetic/<LOG>_sim/` (not in `main.py`) |

`main.py` auto-routes the results root: `*_sim` / `ALL_SIM_LOGS` →
`results/n_grams/sim`, otherwise `results/n_grams/real`. Mixed real+sim names
require an explicit `--output-dir`. Step 06 is not part of the orchestrator;
run it first, then pass `ALL_SIM_LOGS` (or `ACCRE_sim`, …) to `main.py`.

Step 03 model policy: **variants** fit all three (`full_range`, `lower_bounded`, `doubly_bounded`); other n-grams fit **only** `lower_bounded_power_law`.

Step 05 defaults to `lower_bounded_power_law` (`--power-law-model` to change).

`--test` writes attachments to `results/test/attachments/` and describe/Clauset/compose
to `results/test/n_grams/` (including `log_info.csv` / `.tex`). That smoke path uses
the real `TEST_BPIC12` log. It is not the synthetic Zipf E2E.

Synthetic Zipf data (known type frequencies, log name `TEST`) is a unittest,
not a `main.py` flag:

```bash
python -m unittest tests.test_pl_synthetic
# E2E via CLI (optional --n-bootstraps; default 1000)
python tests/test_pl_synthetic.py --e2e-test --n-bootstraps 2500
# slow GOF-implementation test (Type I error; not default CI)
RUN_PL_GOF_TEST=1 python -m unittest tests.test_pl_synthetic.SyntheticPLGofImplementationTests
python tests/test_pl_synthetic.py --gof-test --types 30 50 100 1000
# cluster: E2E + GOF-implementation (edit CONFIG in the script)
mkdir -p .slurm/logs
sbatch .slurm/test_pl_synthetic.sh
```

Attachments land under `results/test/attachments/variants/TEST/`; Clauset CSVs
under `results/test/n_grams/`. E2E reuses existing attachments if that file is
already present.

GOF Monte Carlo outputs land under `results/test/gof_implementation/`:
- per-types replicates: `gof_reps_types30.csv`, `gof_reps_types50.csv`, ...
- per-types summaries: `gof_summary_types30.csv`, `gof_summary_types50.csv`, ...
- combined summary: `gof_summary.csv` (one row per type count)

The GOF CLI skips recomputation when both per-types files already exist; use
`--no-skip-existing` to force reruns.

The Slurm script runs as an array with one type count per node
(`30, 50, 100, 1000`). Keep `#SBATCH --array` equal to `${#TYPES[@]}-1`.
`N_BOOTSTRAPS` is passed to both the E2E (`--e2e-test --n-bootstraps`) and the
GOF Monte Carlo.

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
python analyses/n_grams/main.py --combine --output-dir results/n_grams/real
```

Slurm (from repo root). Set `SOURCE=real` or `SOURCE=sim` in both scripts
before submitting:

```bash
mkdir -p .slurm/logs
sbatch .slurm/n_grams_parallel_array.sh   # one task per log (01-03)
# ... wait until array finishes ...
sbatch .slurm/n_grams_combine.sh          # --combine: 04 then 05

# One array task per log: generate *_sim, then --parallel (no BPIC15_*)
sbatch .slurm/n_grams_sim.sh
# ... wait until array finishes, then SOURCE=sim in n_grams_combine.sh ...
sbatch .slurm/n_grams_combine.sh

# Serial pipeline on the DFG-simulated ACCRE log only
sbatch .slurm/n_grams_accre_sim.sh
```
