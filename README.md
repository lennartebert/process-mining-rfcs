# Process Mining RFCs and Power Law Statistics

Author: Lennart Ebert (lennart.ebert@hu-berlin.de).

Utilities and CLI workflows for rank-frequency-curve (RFC) analysis and
power-law testing of process event logs. Code and derived results are under
CC BY 4.0 (`LICENSE.txt`). Included event logs keep their source licenses.

## Installation

```bash
conda env create -f environment.yml
conda activate process-mining-rfcs
```

Use the `process-mining-rfcs` Jupyter kernel for notebooks. Run commands from
the repository root. This environment is enough for the analyses. Rebuilding
the power-law online appendix PDF is optional and needs TeX Live; see
[`analyses/powerlaw_statistics/README.md`](analyses/powerlaw_statistics/README.md).

Shared CLIs:

```bash
python cli/log_info.py --help
python cli/extract_attachments.py --help
python cli/clauset_power_law.py --help
```

`data/full_data_dictionary_template.json` is a pre-populated template for the
24-log benchmark. Copy it to `data/data_dictionary.json` before running.
Download logs that are not already in this repository into the paths listed
in the template. Sources are below.

If `results/attachments/.../attachments.csv.gz` already exists, extraction is
skipped by default.

## Analyses

This repository contains analyses supporting the following papers. Per-analysis
details live in each folder’s README.

### Rank-frequency curves (RFCs)

Folder: [`analyses/rfcs/`](analyses/rfcs/). Details:
[`analyses/rfcs/README.md`](analyses/rfcs/README.md).

Lennart Ebert, Kate Revoredo, Jan Mendling, and Benoît Depaire. 2026.
Understanding Process Dynamics through Trace Variant Rank-Frequency Curves.
In Workshop on Change, Drift, and Dynamics of Organizational Processes (ProDy)
at the International Conference on Business Process Management (BPM Workshops
2026) (Accepted for publication).

Interactive simulation notebook (permalink copy at the repository root, also
under `analyses/rfcs/`):

```bash
jupyter notebook interactive_simulation_experiments.ipynb
jupyter notebook analyses/rfcs/interactive_simulation_experiments.ipynb
jupyter notebook analyses/rfcs/preset_simulation_experiments.ipynb
jupyter notebook analyses/rfcs/analyze_powerlaw_single_log.ipynb
```

### Power-law statistics

Folder: [`analyses/powerlaw_statistics/`](analyses/powerlaw_statistics/).
Details: [`analyses/powerlaw_statistics/README.md`](analyses/powerlaw_statistics/README.md).

ICPM 2027 submission: Assessing Power-Law Compatibility of Process Behavior
Frequencies Across Orders of Behavioral Abstraction. The online appendix is
`results/perm/powerlaw_statistics/Online Appendix Powerlaw Compatibility.pdf`.

```bash
python analyses/powerlaw_statistics/main.py --datasets BPIC12
python analyses/powerlaw_statistics/main.py --test
```

## Repository structure

```text
data/                 # event logs, data_dictionary.json, images/
results/              # generated outputs (perm/ = git-tracked appendix)
utils/                # reusable library (no CLI entrypoints)
cli/                  # shared CLIs only (import utils/)
  log_info.py
  extract_attachments.py
  clauset_power_law.py
analyses/
  rfcs/               # notebooks (interactive copy also at repo root)
  powerlaw_statistics/# 01 describe -> 02 extract -> 03 Clauset -> 05 cross-log -> 06 compose -> 07 sensitivity
tests/
interactive_simulation_experiments.ipynb  # permalink copy of analyses/rfcs/
```

```text
results/
  attachments/<concept>/<log>/attachments.csv.gz
  rfcs/experiments/…
  rfcs/<log>/…
  pl_stats/<experiment>/log_info/
  pl_stats/<experiment>/<xmin>/raw_results/<n001|variants>/<model>/…
  pl_stats/<experiment>/<xmin>/clauset_results/min[30]-gofp[0.10]-pcomp[0.10]/
  pl_stats/<experiment>/<xmin>/raw_results/plots/<log>/…
  test/…
  perm/rfcs/…
  perm/powerlaw_statistics/…
```

`results/perm/` holds git-tracked permanent copies (manual). Other paths under
`results/` are generated and gitignored.

Dependency rule: analysis step scripts do not import each other. They may call shared `cli/` tools and `utils/`. Per-analysis `main.py` files only orchestrate ordered steps.

## Data

### Included Datasets

This repository includes the following datasets ready to use:

**Real-World Event Logs:**
- **BPIC12 (BPI Challenge 2012)**: van Dongen, B. (2012). BPI Challenge 2012 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
- **RTFMP (Road Traffic Fine Management Process)**: de Leoni, M., & Mannhardt, F. (2015). Road Traffic Fine Management Process (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5
- **TEST_BPIC12**: small BPIC12 sample for `analyses/powerlaw_statistics/main.py --test`. Not a study log.

### Not included datasets

The pipeline supports all datasets listed in `utils/constants.py` (`ALL_REAL_LOG_DATASETS`). The following are **not** shipped in this repository but can be obtained from their original sources:

**Real-World Event Logs:**
- **ACCRE (Academic Credentialing)**: Chapela-Campa, D., Benchekroun, I., Baron, O., Dumas, M., Krass, D., & Senderovich, A. (2024). Evaluation datasets and results of the paper "A Framework for Measuring the Quality of Business Process Simulation Models" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12126071
- **BPIC11 (BPI Challenge 2011, Dutch Academic Hospital)**: van Dongen, B. (2011). Real-life event logs - Hospital log (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54
- **BPIC13_cp (BPI Challenge 2013, closed problems)**: Steeman, W. (2013). BPI Challenge 2013, closed problems (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11
- **BPIC13_i (BPI Challenge 2013, incidents)**: Steeman, W. (2013). BPI Challenge 2013, incidents (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee
- **BPIC14 (BPI Challenge 2014)**: van Dongen, B. (2014). BPI Challenge 2014: Incident details (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:3cfa2260-f5c5-44be-afe1-b70d35288d6d
- **BPIC15_1 (BPI Challenge 2015, Municipality 1)**: van Dongen, B. F. (2015). BPI Challenge 2015 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1
- **BPIC15_2 (BPI Challenge 2015, Municipality 2)**: van Dongen, B. F. (2015). BPI Challenge 2015 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1
- **BPIC15_3 (BPI Challenge 2015, Municipality 3)**: van Dongen, B. F. (2015). BPI Challenge 2015 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1
- **BPIC15_4 (BPI Challenge 2015, Municipality 4)**: van Dongen, B. F. (2015). BPI Challenge 2015 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1
- **BPIC15_5 (BPI Challenge 2015, Municipality 5)**: van Dongen, B. F. (2015). BPI Challenge 2015 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1
- **BPIC17 (BPI Challenge 2017)**: van Dongen, B. (2017). BPI Challenge 2017 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
- **BPIC18 (BPI Challenge 2018, EU Agriculture Subsidy)**: van Dongen, B., & Borchert, F. (2018). BPI Challenge 2018 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972
- **BPIC19 (BPI Challenge 2019)**: van Dongen, B. (2019). BPI Challenge 2019 (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1
- **BPIC20_dd (BPI Challenge 2020, Domestic Declarations)**: van Dongen, B. (2020). BPI Challenge 2020: Domestic Declarations (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5
- **BPIC20_id (BPI Challenge 2020, International Declarations)**: van Dongen, B. (2020). BPI Challenge 2020: International Declarations (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5
- **BPIC20_ptc (BPI Challenge 2020, Prepaid Travel Costs)**: van Dongen, B. (2020). BPI Challenge 2020: Prepaid Travel Costs (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165
- **BPIC20_rfp (BPI Challenge 2020, Request For Payment)**: van Dongen, B. (2020). BPI Challenge 2020: Request For Payment (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030
- **BPIC20_tpd (BPI Challenge 2020, Travel Permit Data)**: van Dongen, B. (2020). BPI Challenge 2020: Travel Permit Data (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550
- **CALL**: Chapela-Campa, D., Benchekroun, I., Baron, O., Dumas, M., Krass, D., & Senderovich, A. (2024). Evaluation datasets and results of the paper "A Framework for Measuring the Quality of Business Process Simulation Models" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12126071
- **ITHD (Help Desk)**: Polato, M. (2017). Dataset belonging to the help desk log of an Italian Company (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb
- **MOBIS (MobIS Challenge 2019)**: Scheid, M., Rehse, J.-R., Houy, C., & Fettke, P. (2018). Data Set for MobIS Challenge 2019 [Data set]. German Research Center for Artificial Intelligence (DFKI). https://www.dfki.de/web/forschung/projekte-publikationen/publikation/10258
- **SEPSIS (Sepsis Cases)**: Mannhardt, F. (2016). Sepsis Cases - Event Log (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460

**Synthetic Event Logs:**
- **CVS (pharmacy)**: Camargo, M., Dumas, M., & González-Rojas, O. (2021). Learning Accurate Business Process Simulation Models from Event Logs via Automated Process Discovery and Deep Learning (Version 0.0.3). Zenodo. https://doi.org/10.5281/zenodo.5734443

### Preprocessing

Most real logs are used as published XES (`.xes.gz`). CSV sources (and a BPIC12 smoke-test sample) are converted by [`analyses/01_preprocessing.ipynb`](analyses/01_preprocessing.ipynb). Analysis-time import and activity labeling are shared across all logs:

- **Import / ordering.** Logs are loaded via `utils/io/event_logs.py` (`get_event_log_from_path`). By default, events are sorted by `time:timestamp` ascending (`timestamp_sort=True`; used by `log_info` and `extract_attachments`). Traces are then ordered by their first event’s timestamp. **Equal timestamps keep their relative order from the XES file** (Python/PM4Py stable sort; no secondary key). CSV→XES conversions also globally sort by timestamp before export.
- **Activity label.** After import, `utils/io/activity_labels.py` builds labels for attachments, variants, and `log_info`: if the log defines an XES classifier named **`Activity classifier`**, its keys are joined with `+` (e.g. `concept:name` + `lifecycle:transition` → `A_SUBMITTED+COMPLETE`); otherwise the activity name only (`concept:name`, or the CSV activity column mapped to it). Lifecycle may be present on events without being part of the label (e.g. BPIC17).

| Log | Source | Format | Case definition | Activity name column | Activity classifier | Lifecycle transition available | Activity label | Timestamp | Ordering | Filtering | Verification |
|-----|--------|--------|-----------------|----------------------|---------------------|--------------------------------|----------------|-----------|----------|-----------|--------------|
| ACCRE | `AcademicCredentials_train.csv.gz` (Zenodo; train only) | CSV | `case_id` | `activity` | — | No | activity name | `start_time` | Timestamp sort; ties → XES parse order | Train only (exclude test; avoid train/test boundary gaps) | Conversion verified |
| BPIC11 | `hospital_log.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — (has `Event Name`: `concept:name`) | Yes | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (exact) |
| BPIC12 | `BPI_Challenge_2012.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | `concept:name` + `lifecycle:transition` | Yes | activity name + lifecycle transition | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC13_cp | `BPI_Challenge_2013_closed_problems.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | `concept:name` + `lifecycle:transition` | Yes | activity name + lifecycle transition | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC13_i | `BPI_Challenge_2013_incidents.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | `concept:name` + `lifecycle:transition` | Yes | activity name + lifecycle transition | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC14 | `Detail_Incident_Activity.csv.gz` (`;`) | CSV | `Incident ID` | `IncidentActivity_Type` | — | No | activity name | `DateStamp` | Timestamp sort; ties → XES parse order | None | Conversion verified |
| BPIC15_1 | `BPIC15_1.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | `concept:name` + `lifecycle:transition` | Yes | activity name + lifecycle transition | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC15_2 | `BPIC15_2.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | `concept:name` + `lifecycle:transition` | Yes | activity name + lifecycle transition | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC15_3 | `BPIC15_3.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | `concept:name` + `lifecycle:transition` | Yes | activity name + lifecycle transition | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC15_4 | `BPIC15_4.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | `concept:name` + `lifecycle:transition` | Yes | activity name + lifecycle transition | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC15_5 | `BPIC15_5.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | `concept:name` + `lifecycle:transition` | Yes | activity name + lifecycle transition | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC17 | `BPI Challenge 2017.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — (has `Event Name`: `concept:name`; `MXML Legacy Classifier`: `concept:name` + `lifecycle:transition`) | Yes | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC18 | `BPI Challenge 2018.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — (has `Event Name`: `concept:name`) | Yes | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC19 | `BPI_Challenge_2019.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — (has `Event Name`: `concept:name`) | No | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC20_dd | `DomesticDeclarations.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — | No | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (exact) |
| BPIC20_id | `InternationalDeclarations.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — | No | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC20_ptc | `PrepaidTravelCost.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — | No | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC20_rfp | `RequestForPayment.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — | No | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| BPIC20_tpd | `PermitLog.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — | No | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| CALL | `CallCenter_train.csv.gz` (Zenodo; train only) | CSV | `case_id` | `activity` | — | No | activity name | `end_time` | Timestamp sort; ties → XES parse order | Train only (same rationale as ACCRE) | Conversion verified |
| ITHD | `finale.csv.gz` | CSV | `Case ID` | `Activity` | — | No | activity name | `Complete Timestamp` | Timestamp sort; ties → XES parse order | None | Conversion verified |
| MOBIS | `10258_mobis_challenge_log_2019.csv.gz` (`;`) | CSV | `case` | `activity` | — | No | activity name | `end` | Timestamp sort; ties → XES parse order | None | Conversion verified |
| RTFMP | `Road_Traffic_Fine_Management_Process.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — (has `Event Name`: `concept:name`) | Yes | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |
| SEPSIS | `Sepsis Cases - Event Log.xes.gz` (published) | XES | `concept:name` (trace) | `concept:name` | — (has classifier `concept:name`: `concept:name`) | Yes | activity name | `time:timestamp` | Timestamp sort; ties → XES parse order | None | Published source match completed (uncompressed XES) |

**Equal timestamps.** Sorting uses only `time:timestamp`. When two events (or two traces’ first events) share the same timestamp, their relative order is left unchanged from the order in the loaded XES (parse / file order). There is no secondary tie-break key (e.g. activity name).
