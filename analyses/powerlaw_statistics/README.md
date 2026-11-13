# Power-law statistics analysis

Pipeline for describing logs, extracting variant + n-gram (n1..n20) attachments,
running Clauset-style power-law tests, combining parallel shards, cross-log
analyses, and composing summary tables/plots. Shared tools live under `cli/`.
Analysis libraries live under `analysis_helpers/`. Synthetic generators and the
inspection notebook live under `exploratory/`.

Describe / Clauset / compose outputs go under `results/pl_stats/real` for real
logs, `results/pl_stats/synth_dfg` for `*_sim` DFG replays, and
`results/pl_stats/synth` for model/monkey synthetics. A different bootstrap seed
is a different experiment name (`real43`). xmin is the next folder
(`xmin-adaptive`, `xmin01`, `xmin02`). Attachments stay under
`results/attachments/<concept>/<log>/`.

```bash
# Full serial pipeline 01 -> 02 -> 03 -> 05 -> 06 (skips combine)
python analyses/powerlaw_statistics/main.py --datasets TEST_BPIC12

# All real benchmark logs (writes results/pl_stats/real)
python analyses/powerlaw_statistics/main.py --datasets ALL_REAL_LOGS

# First-order DFG-simulated counterparts (writes results/pl_stats/synth_dfg)
python analyses/powerlaw_statistics/exploratory/generate_synthetic_from_real_logs.py
python analyses/powerlaw_statistics/main.py --datasets ALL_SIM_LOGS

# Iid monkey-typing synthetic logs
python analyses/powerlaw_statistics/exploratory/generate_synthetic_monkey.py \
  --n-activities 26 --n-traces 20000 --trace-length 20
python analyses/powerlaw_statistics/exploratory/generate_synthetic_monkey.py \
  --n-activities 10 --n-traces 20000 --trace-length 20
python analyses/powerlaw_statistics/exploratory/generate_synthetic_monkey.py \
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
| 01 | `01_describe.py` | Event-log description via `cli/log_info.py` → `<experiment>/log_info/` |
| 02 | `02_extract_ngrams.py` | Attachments under `results/attachments/<concept>/<log>/` |
| 03 | `03_clauset_tests.py` | Clauset fits under `<experiment>/<xmin>/raw_results/<n001\|variants>/<model>/` |
| 04 | `04_combine_parallel.py` | Merge parallel `log_info` + Clauset shards only |
| 05 | `05_cross_log_analysis.py` | Saturation, paired GOF, and GOF-change exploration from saved results |
| 06 | `06_compose_results.py` | A–E tables under `<xmin>/clauset_results/min[30]-gofp[0.10]-pcomp[0.10]/`; plots under `raw_results/plots/` |
| 07 | `07_cross_configuration_analysis.py` | Sensitivity across xmin / thresholds (not in `--combine`) |

Standalone (not in `main.py` `STEPS`):

| Script | Role |
|--------|------|
| `exploratory/generate_synthetic_from_real_logs.py` | DFG simulator from real logs → `data/synthetic/<LOG>_sim/` |
| `exploratory/generate_synthetic_monkey.py` | Iid monkey-typing synthetic log → `data/synthetic/MONKEY_*/` |

`main.py` auto-routes the experiment directory: `*_sim` / `ALL_SIM_LOGS` →
`results/pl_stats/synth_dfg`, otherwise `results/pl_stats/real`.
Model/monkey logs need `--output-dir results/pl_stats/synth` (or Slurm `SOURCE=synth`).
`--xmin 1` writes `xmin01` inside that experiment; the default is `xmin-adaptive`.
Mixed real+sim names require an explicit `--output-dir`. Generate synthetic
logs first, then pass `ALL_SIM_LOGS` (or `ACCRE_sim`, …) to `main.py`.

Step 03 model policy: **variants** fit all three (`full_range`, `lower_bounded`, `doubly_bounded`); other n-grams fit **only** `lower_bounded_power_law`.

Step 06 defaults to `lower_bounded_power_law` (`--power-law-model` to change).

`--test` writes attachments to `results/test/attachments/` and describe/Clauset/cross-log/compose
to `results/test/pl_stats/` (`log_info/` and `xmin-adaptive/raw_results/`). That smoke path uses
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
under `results/test/pl_stats/`. E2E reuses existing attachments if that file is
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
to execute **04 → 05 → 06** (merge shards, cross-log, compose). The combine
sbatch does the same:

```bash
sbatch .slurm/powerlaw_statistics_combine.sh real
```

| Writer | Shard | After `--combine` (step 04) |
|--------|-------|-----------------------------|
| describe | `log_info_<LOG>.csv` | `log_info.csv` + `.tex` |
| Clauset | `<concept>/<model>/{gof,comparison,summary}_<LOG>.csv` | unsuffixed CSVs |

Step 05 writes cross-log products for real-log experiments:

- saturation under `<experiment>/log_info/saturation/`
- paired GOF under `<xmin>/clauset_results/min[30]-gofp[0.10]-pcomp[0.10]/` (defaults)
- GOF-change exploration under `<xmin>/explore_gof_changes/<LOG>/`

Step 06 then builds `variant_power_law.*`, `log_n_scaling.*`, `pl_outcome_summary.*`, plots, etc. from the combined summaries. It can later read step-05 artifacts for LaTeX.

Paired GOF reuses saved summary rows and does not refit or rerun GOF. Outputs:

- `paired_gof_summary.csv` — one row per k = 1, ..., 20 versus trace variants
- `paired_gof_audit.csv` — one row per study log and k
- `paired_gof_paper_table.csv` — the wide table, one column per k

The xmin policy is the folder name (`xmin-adaptive`, `xmin01`, `xmin02`), not the fitted xmin: an adaptive fit that selected xmin = 1 stays in `xmin-adaptive`. Variants come from that same folder. Comparison p does not enter the paired outcome; the files are still stored in the configuration folder that uses it. Missing results are reported rather than replaced.

A log is paired only when both endpoints are eligible. Eligibility is a valid saved fit/GOF result with at least the stated number of fitted types. It is recalculated for each cutoff and is not read from the A–E labels. The outcome is GOF non-rejection, `gof_p >= threshold`, with `gof_p` kept as a probability. The reported difference is the non-rejection proportion at trace variants minus the proportion at k. The 95% interval is Newcombe's paired method 10 (PMID 9839354) and is pointwise. The two-sided exact McNemar p-values are Holm-adjusted within the k = 1, ..., 20 family written for that configuration.

The paper table keeps one column per k. Difference and CI bounds are the probability-scale values multiplied by 100 once. If a comparison has no jointly eligible logs, `paired_N` is 0 and the estimates and p-values are `NaN`. A comparison with no discordant pairs keeps raw McNemar p = 1. Missing source results stay in their column; the combine log states why. Three Holm rows are written from the raw p-values, each with its own family: k <= 20, k <= 10, and k <= 3. Columns outside a family are `NaN`. These rows are alternative family definitions, not three independent confirmations. The paper has to justify which family supports the main inference rather than choosing the family whose adjustment is significant.

The pairing unit is the log, and logs are treated as independent. Related benchmarks and the non-random study collection limit generalisation. Non-rejection does not show that a power law is true or best, and a non-significant difference does not show equivalence. The intervals describe differences across logs, not Monte Carlo error in one GOF p-value.

## Cross-configuration sensitivity

Step 07 recalculates the one-at-a-time grid from saved adaptive results across
thresholds. It does not refit, rerun likelihood-ratio tests, or rerun GOF
bootstraps. It is **not** part of `--combine`.

```bash
python analyses/powerlaw_statistics/07_cross_configuration_analysis.py
```

Outputs land in `results/pl_stats/real/sensitivity/`
(`configurations.csv` / `.tex` and `sensitivity_summary.csv` / `.tex`).
Fits and bootstraps are not rerun. `--experiment` defaults to
`results/pl_stats/real`.

## Saturation

Step 05 writes cross-log saturation tables for real-log experiments (directory name starting with `real`), under `<experiment>/log_info/saturation/`:

- `saturation_summary.csv` — one row per study log and order k in {1, 5, 10, 15, 20}, with shares on the probability scale
- `k1.csv`, `k5.csv`, `k10.csv`, `k15.csv`, `k20.csv` — one table per order, shares in percentage points, with an unweighted median row
- `cross_log_saturation.csv` — one row per study log (case-completion percent at k = 5, 10, 15, 20, plus the k = 20 complete-case occurrence share and excess occurrences per case) and summary rows for logs at or above 50%, 60%, 70%, 80%, 90%, and 100% case completion

The counts use every study log. They do not depend on xmin, the fitted-type cutoff, or the GOF threshold, and they are not read from the GOF tables. Trace length excludes the START and END markers. A case is complete when it contributes one occurrence: at k = 1 that is length 1 (n1 does not wrap), and at k >= 2 it is length plus 2 at most k. Each case contributes L occurrences at k = 1 and `max(1, L + 3 - k)` at k >= 2, which is the extractor's window count. The median row is the unweighted median across logs. Logs without a usable trace set are left blank and are not part of the summary-row denominators. Formula totals are checked against the saved occurrence totals wherever those totals exist.

```bash
# One log (local or Slurm array task)
python analyses/powerlaw_statistics/main.py --datasets BPIC12 --parallel

# Smoke test in parallel mode
python analyses/powerlaw_statistics/main.py --test --parallel

# After all shards exist: merge + cross-log + compose
python analyses/powerlaw_statistics/main.py --combine --output-dir results/pl_stats/real
```

Slurm (from repo root). Pass `SOURCE` as a positional arg (`real` default;
also `synth` = model/monkey, `synth_dfg` = `*_sim`):

```bash
mkdir -p .slurm/logs
sbatch .slurm/powerlaw_statistics_parallel_array.sh                     # real
sbatch .slurm/powerlaw_statistics_parallel_array.sh synth               # O2C_S / monkey / …
# ... wait until array finishes ...
sbatch .slurm/powerlaw_statistics_combine.sh synth                      # same SOURCE; runs 04→05→06

# One array task per log: generate *_sim, then --parallel (no BPIC15_*)
sbatch .slurm/powerlaw_statistics_sim.sh
# ... wait until array finishes ...
sbatch .slurm/powerlaw_statistics_combine.sh synth_dfg

# Serial pipeline on the DFG-simulated ACCRE log only
sbatch .slurm/powerlaw_statistics_accre_sim.sh
```
