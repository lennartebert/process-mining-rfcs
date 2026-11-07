# Power-law statistics analysis

Pipeline for describing logs, extracting variant + n-gram (n1..n20) attachments,
running Clauset-style power-law tests, combining parallel shards, and composing
summary tables/plots. Shared tools live under `cli/`.

Describe / Clauset / compose outputs go under `results/powerlaw_statistics/real` for real
logs, `results/powerlaw_statistics/sim_from_real_dfgs` for `*_sim` DFG replays, and
`results/powerlaw_statistics/sim` for model/monkey synthetics. Attachments stay under
`results/attachments/<concept>/<log>/`.

```bash
# Full serial pipeline 01 -> 02 -> 03 -> 05 (skips combine)
python analyses/powerlaw_statistics/main.py --datasets TEST_BPIC12

# All real benchmark logs (writes results/powerlaw_statistics/real)
python analyses/powerlaw_statistics/main.py --datasets ALL_REAL_LOGS

# First-order DFG-simulated counterparts (writes results/powerlaw_statistics/sim_from_real_dfgs)
python analyses/powerlaw_statistics/06_generate_synthetic_from_real_logs.py
python analyses/powerlaw_statistics/main.py --datasets ALL_SIM_LOGS

# Iid monkey-typing synthetic logs
python analyses/powerlaw_statistics/07_generate_synthetic_monkey.py \
  --n-activities 26 --n-traces 20000 --trace-length 20
python analyses/powerlaw_statistics/07_generate_synthetic_monkey.py \
  --n-activities 10 --n-traces 20000 --trace-length 20
python analyses/powerlaw_statistics/07_generate_synthetic_monkey.py \
  --n-activities 24 --n-traces 20000 --trace-length 40

# Restrict concepts (default: n1..n20 + variants)
python analyses/powerlaw_statistics/main.py --datasets TEST_BPIC12 --concepts n1 n2 variants

# Quick smoke test (TEST_BPIC12; concepts n1, n2, variants; fewer bootstraps)
python analyses/powerlaw_statistics/main.py --test

# Subset
python analyses/powerlaw_statistics/main.py --datasets TEST_BPIC12 --from 02 --to 03
python analyses/powerlaw_statistics/main.py --datasets TEST_BPIC12 --only 05
```

| Step | Script | Role |
|------|--------|------|
| 01 | `01_describe.py` | Event-log description via `cli/log_info.py` → `results/powerlaw_statistics/real/log_info.*` |
| 02 | `02_extract_ngrams.py` | Attachments under `results/attachments/<concept>/<log>/` |
| 03 | `03_clauset_tests.py` | Clauset fits under `results/powerlaw_statistics/real/<concept>/<model>/` |
| 04 | `04_combine_parallel.py` | Merge parallel `log_info` + Clauset shards (after `--parallel`) |
| 05 | `05_compose_results.py` | Tables + plots under `results/powerlaw_statistics/real/` |
| 06 | `06_generate_synthetic_from_real_logs.py` | Standalone DFG simulator from real logs → `data/synthetic/<LOG>_sim/` (not in `main.py`) |
| 07 | `07_generate_synthetic_monkey.py` | Iid monkey-typing synthetic log → `data/synthetic/MONKEY_*/` (not in `main.py`) |

`main.py` auto-routes the results root: `*_sim` / `ALL_SIM_LOGS` →
`results/powerlaw_statistics/sim_from_real_dfgs`, otherwise `results/powerlaw_statistics/real`.
Model/monkey logs need `--output-dir results/powerlaw_statistics/sim` (or Slurm `SOURCE=sim`).
Mixed real+sim names require an explicit `--output-dir`. Step 06 is not part of the
orchestrator; run it first, then pass `ALL_SIM_LOGS` (or `ACCRE_sim`, …) to `main.py`.

Step 03 model policy: **variants** fit all three (`full_range`, `lower_bounded`, `doubly_bounded`); other n-grams fit **only** `lower_bounded_power_law`.

Step 05 defaults to `lower_bounded_power_law` (`--power-law-model` to change).

`--test` writes attachments to `results/test/attachments/` and describe/Clauset/compose
to `results/test/powerlaw_statistics/` (including `log_info.csv` / `.tex`). That smoke path uses
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
under `results/test/powerlaw_statistics/`. E2E reuses existing attachments if that file is
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

Step 04 also writes paired GOF comparisons when the results directory is one of the real-log xmin experiments (`real`, `real_xmin1`, `real_xmin2`). It reuses saved summary rows and does not refit or rerun GOF. Outputs land in `<output-dir>/paired_gof/`:

- `paired_gof_summary.csv` — one row per sensitivity configuration and k-versus-variants comparison
- `paired_gof_audit.csv` — one row per study log, configuration, and comparison
- `paired_gof_results.txt` — the same comparisons in percentage points, with the methods note
- `paired_gof_paper_table.csv` — k = 1, ..., 20 against trace variants at adaptive xmin, 30 fitted types, and GOF threshold 0.10

The comparisons are k = 1, 2, 3, and 20 against trace variants, on each combination of `min_fitted_types` in {15, 30, 50}, GOF threshold in {0.05, 0.10}, and xmin experiment in {fixed 1, fixed 2, adaptive}. The xmin experiment is the results directory, not the fitted xmin: an adaptive fit that selected xmin = 1 stays in `real`. Variants come from that same directory. Missing results are reported rather than replaced.

A log is paired only when both endpoints are eligible. Eligibility is a valid saved fit/GOF result with at least the stated number of fitted types. It is recalculated for each cutoff and is not read from the A–E labels. The outcome is GOF non-rejection, `gof_p >= threshold`, with `gof_p` kept as a probability. The reported difference is the non-rejection proportion at trace variants minus the proportion at k. The 95% interval is Newcombe's paired method 10 (PMID 9839354) and is pointwise. The two-sided exact McNemar p-values are Holm-adjusted across the five comparisons within each configuration. All 18 configurations are exported.

The paper table keeps one column per k. Difference and CI bounds are the probability-scale values multiplied by 100 once. If a comparison has no jointly eligible logs, `paired_N` is 0 and the estimates and p-values are `NaN`. A comparison with no discordant pairs keeps raw McNemar p = 1. Missing source results stay in their column; the combine log states why. Three Holm rows are written from the raw p-values, each with its own family: k <= 20, k <= 10, and k <= 3. Columns outside a family are `NaN`. These rows are alternative family definitions, not three independent confirmations. The paper has to justify which family supports the main inference rather than choosing the family whose adjustment is significant.

The pairing unit is the log, and logs are treated as independent. Related benchmarks and the non-random study collection limit generalisation. Non-rejection does not show that a power law is true or best, and a non-significant difference does not show equivalence. The intervals describe differences across logs, not Monte Carlo error in one GOF p-value.

Step 04 also writes cross-log saturation tables for those same real-log directories, under `<output-dir>/saturation/`:

- `saturation_summary.csv` — one row per study log and order k in {5, 10, 15, 20}, with shares on the probability scale
- `saturation_results.txt` — one readable table per order

The counts use every study log. They do not depend on xmin, the fitted-type cutoff, or the GOF threshold, and they are not read from the GOF tables. Trace length excludes the START and END markers. A case or variant is complete when its length plus 2 is at most k. Each case contributes `max(1, L + 3 - k)` occurrences, which is the extractor's window count. The median row is the unweighted median across logs. Formula totals are checked against the saved occurrence totals wherever those totals exist.

```bash
# One log (local or Slurm array task)
python analyses/powerlaw_statistics/main.py --datasets BPIC12 --parallel

# Smoke test in parallel mode
python analyses/powerlaw_statistics/main.py --test --parallel

# After all shards exist: merge + compose
python analyses/powerlaw_statistics/main.py --combine --output-dir results/powerlaw_statistics/real
```

Slurm (from repo root). Pass `SOURCE` as a positional arg (`real` default;
also `sim` = model/monkey, `sim_from_real_dfgs` = `*_sim`):

```bash
mkdir -p .slurm/logs
sbatch .slurm/powerlaw_statistics_parallel_array.sh                     # real
sbatch .slurm/powerlaw_statistics_parallel_array.sh sim                 # O2C_S / monkey / …
# ... wait until array finishes ...
sbatch .slurm/powerlaw_statistics_combine.sh sim                        # same SOURCE

# One array task per log: generate *_sim, then --parallel (no BPIC15_*)
sbatch .slurm/powerlaw_statistics_sim.sh
# ... wait until array finishes ...
sbatch .slurm/powerlaw_statistics_combine.sh sim_from_real_dfgs

# Serial pipeline on the DFG-simulated ACCRE log only
sbatch .slurm/powerlaw_statistics_accre_sim.sh
```
