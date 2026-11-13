# Power-law statistics

Clauset-style power-law tests of trace-variant and n-gram (n1..n20) frequencies
for the ICPM 2027 submission *Assessing Power-Law Compatibility of Process
Behavior Frequencies Across Orders of Behavioral Abstraction*.

Run commands from the repository root. Copy a data dictionary to
`data/data_dictionary.json` first: `data/default_data_dictionary.json` for the
shipped logs, or `data/full_data_dictionary_template.json` after you have
placed the 24 benchmark logs at the paths in that file. Those 24 logs are not
in this repository. Sources are in the repository README.

```bash
# Serial real-log pipeline: 01 -> 02 -> 03 -> 05 -> 06 -> 07
python analyses/powerlaw_statistics/main.py --datasets BPIC12

# 24-log benchmark
python analyses/powerlaw_statistics/main.py --datasets ALL_REAL_LOGS

# Smoke test (TEST_BPIC12; concepts n1, n2, variants; 50 bootstraps; no step 07)
python analyses/powerlaw_statistics/main.py --test

# One log at a time (01 -> 02 -> 03), then merge, cross-log, compose, sensitivity
python analyses/powerlaw_statistics/main.py --datasets BPIC12 --parallel
python analyses/powerlaw_statistics/main.py --combine --output-dir results/pl_stats/real

python analyses/powerlaw_statistics/main.py --datasets BPIC12 --from 02 --to 03
python analyses/powerlaw_statistics/main.py --only 07
```

| Step | Script | Role |
|------|--------|------|
| 01 | `01_describe.py` | Log description → `<experiment>/log_info/` |
| 02 | `02_extract_ngrams.py` | Attachments → `results/attachments/<concept>/<log>/` |
| 03 | `03_clauset_tests.py` | Clauset fits → `<experiment>/<xmin>/raw_results/` |
| 04 | `04_combine_parallel.py` | Merge per-log shards written by `--parallel` |
| 05 | `05_cross_log_analysis.py` | Saturation, paired GOF, and GOF-change exploration |
| 06 | `06_compose_results.py` | Summary tables and plots |
| 07 | `07_cross_configuration_analysis.py` | Threshold sensitivity from saved adaptive fits |

Default concepts are n1..n20 and trace variants. `--concepts` narrows steps
02, 03, 04, and 06. Step 03 fits `full_range`, `lower_bounded`, and
`doubly_bounded` on variants, and only `lower_bounded_power_law` on n-grams.
Step 06 reads `lower_bounded_power_law`.

Real logs write to `results/pl_stats/real`. `*_sim` logs write to
`results/pl_stats/synth_dfg`. Other synthetics need
`--output-dir results/pl_stats/synth`. `--xmin 1` writes `xmin01`; the default
folder is `xmin-adaptive`. A different bootstrap seed belongs in a different
experiment directory, such as `results/pl_stats/real43`. Mixed real and
`*_sim` names need an explicit `--output-dir`.

Step 07 recalculates eligibility, categories, and paired GOF across threshold
settings. It does not refit. On the default path it runs only when the
experiment directory name starts with `real`, and it writes
`<experiment>/sensitivity/`. `--only`, `--from`, and `--to` can still select
it. `--parallel` stops after step 03. `--combine` runs 04 through 07.

Paper outputs already computed for the submission are under
`results/perm/powerlaw_statistics/`.

## Synthetic logs

These generators are not `main.py` steps. They write `data/synthetic/` and
register the new logs in `data/data_dictionary.json`.

```bash
python analyses/powerlaw_statistics/exploratory/generate_synthetic_from_real_logs.py
python analyses/powerlaw_statistics/main.py --datasets ALL_SIM_LOGS

python analyses/powerlaw_statistics/exploratory/generate_synthetic_monkey.py \
  --n-activities 26 --n-traces 20000 --trace-length 20
```

`ALL_SIM_LOGS` is a `*_sim` run, so step 07 is not included unless you select
it.

## Online appendix

The built PDF is already in the repository. Regenerating it is optional and
needs TeX Live, with `latexmk` on `PATH` (`pdflatex` is the fallback). A
medium or full scheme includes the packages in `appendix/main.tex`, including
`xurl` and `pdflscape`. LaTeX is not part of `environment.yml`.

```bash
python analyses/powerlaw_statistics/appendix/generate.py --perm
```

`--perm` reads `results/perm/powerlaw_statistics/` and writes
`results/perm/powerlaw_statistics/Online Appendix Powerlaw Compatibility.pdf`.
Without `--perm`, the PDF is written under `results/pl_stats/`.

## Tests

```bash
python -m unittest tests.test_pl_synthetic
python tests/test_pl_synthetic.py --e2e-test --n-bootstraps 2500
RUN_PL_GOF_TEST=1 python -m unittest tests.test_pl_synthetic.SyntheticPLGofImplementationTests
```

The synthetic Zipf log (`TEST`) is a unit test. It is not `main.py --test`.
