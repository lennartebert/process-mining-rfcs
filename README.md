# RFCs in Process Mining

Author: Lennart Ebert (lennart.ebert@hu-berlin.de).

Utilities and CLI workflows for rank-frequency-curve (RFC) analysis in process mining, including:

- attachment extraction from event logs,
- static RFC/power-law fitting,
- synthetic process simulation experiments
- (under development) dynamic preferential-attachment analysis.

## TL;DR

### Online appendix files
- `results/perm/log_info/log_info.csv`: Key statistics of logs under review
- `results/perm/variants/rfc_static_analysis_summary.csv`: Model fitting results
- `results/perm/experiments/...`: BPMN models and RFC curves of all experiments

### Reproducing the study

```bash
# 1) create environment
conda env create -f environment.yml
conda activate process-mining-rfcs

# 2) n-gram pipeline (describe -> extract variants+n1..n10 -> Clauset tests)
python analyses/n_grams/main.py --datasets TEST_BPIC12
```

More recipes: [`analyses/n_grams/README.md`](analyses/n_grams/README.md).

If you want to (re)calculate attachments from raw logs, place your `.xes` files under `data/`, create a copy of the data dictionary (to be placed `data/data_dictionary.json`) and ensure refer to the data sets in the dictionary.
If `results/.../attachments.csv.gz` already exists, extraction is skipped by default.

You can also run the analysis/simulation notebooks:

```bash
jupyter notebook analyses/rfcs/analyze_powerlaw_single_log.ipynb
jupyter notebook analyses/rfcs/interactive_simulation_experiments.ipynb
jupyter notebook analyses/rfcs/preset_simulation_experiments.ipynb
```

Use the `process-mining-rfcs` Jupyter kernel for these notebooks.

## Repository structure

The repo is structured as follows:

- `data/` contains event-log inputs, dataset metadata, and static assets (`data/images/`).
- `results/` contains generated outputs (attachments, RFC analysis artifacts, and experiment artifacts). Permanent, version-controlled results live under `results/perm/`.
- `utils/` contains reusable library modules (no CLI entrypoints).
- `cli/` contains shared command-line tools only: `log_info.py`, `extract_attachments.py`, `clauset_power_law.py` (import `utils/` only).
- `analyses/` contains topic pipelines and notebooks:
  - `analyses/rfcs/` — notebooks only
  - `analyses/n_grams/` — numbered n-gram / Clauset pipeline + `main.py`
  - `analyses/preferential_attachment/` — preferential-attachment detection
  - `analyses/case_attributes/` — case-attribute pipeline + `main.py`
  - `analyses/others/` — legacy / misc scripts (may be removed later)
- `tests/` contains unit tests.

## Activity definition

For attachment extraction, variant counting, and activity statistics (`log_info`), activities are derived from the event log after import:

- If the log defines an XES **Activity classifier**, each activity label combines all of that classifier’s keys (joined with `+`, e.g. `concept:name` and `lifecycle:transition` → `Register+start`).
- Otherwise, activities are taken from **`concept:name`** only.

The same logic lives in `utils/io/activity_labels.py` and is used by `extract_attachments` and `log_info`.

## Running the CLIs

Shared tools (run from the repository root):

```bash
python cli/log_info.py --help
python cli/extract_attachments.py --help
python cli/clauset_power_law.py --help
```

Analysis pipelines:

```bash
python analyses/n_grams/main.py --help
python analyses/case_attributes/main.py --help
python analyses/preferential_attachment/detect_preferential_attachment.py --help
```

Details: [`analyses/n_grams/README.md`](analyses/n_grams/README.md), [`analyses/case_attributes/README.md`](analyses/case_attributes/README.md), [`analyses/rfcs/README.md`](analyses/rfcs/README.md).

## Data

### Included Datasets

This repository includes the following datasets ready to use:

**Real-World Event Logs:**
- **BPIC12 (BPI Challenge 2012)**: van Dongen, B. 2012. BPI Challenge 2012, Media types: application/x-gzip, text/xml, Eindhoven University of Technology, April 23. (https://doi.org/10.4121/UUID:3926DB30-F712-4394-AEBC-75976070E91F).
- **RTFMP (Road Traffic Fine Management Process)**: de Leoni, M. (Massimiliano); Mannhardt, Felix (2015): Road Traffic Fine Management Process. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5

### Not included datasets

The pipeline supports all datasets listed in `utils/constants.py` (`ALL_REAL_LOG_DATASETS`). The following are **not** shipped in this repository but can be obtained from their original sources:

**Real-World Event Logs:**
- **ACCRE (Academic Credentialing)**: Chapela-Campa, D., Benchekroun, I., Baron, O., Dumas, M., Krass, D., & Senderovich, A. (2024). Evaluation datasets and results of the paper "A Framework for Measuring the Quality of Business Process Simulation Models" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12126071
- **BPIC11 (BPI Challenge 2011, Dutch Academic Hospital)**: van Dongen, Boudewijn (2011): Real-life event logs - Hospital log. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54
- **BPIC13_cp (BPI Challenge 2013, closed problems)**: Ward Steeman (2013): BPI Challenge 2013, closed problems. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11
- **BPIC13_i (BPI Challenge 2013, incidents)**: Ward Steeman (2013): BPI Challenge 2013, incidents. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee
- **BPIC14 (BPI Challenge 2014)**: van Dongen, Boudewijn (2014): BPI Challenge 2014: Incident details. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:3cfa2260-f5c5-44be-afe1-b70d35288d6d
- **BPIC15_1 (BPI Challenge 2015, Municipality 1)**: van Dongen, B. F. 2015. "BPI Challenge 2015," Eindhoven University of Technology. (https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1).
- **BPIC15_2 (BPI Challenge 2015, Municipality 2)**: van Dongen, B. F. 2015. "BPI Challenge 2015," Eindhoven University of Technology. (https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1).
- **BPIC15_3 (BPI Challenge 2015, Municipality 3)**: van Dongen, B. F. 2015. "BPI Challenge 2015," Eindhoven University of Technology. (https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1).
- **BPIC15_4 (BPI Challenge 2015, Municipality 4)**: van Dongen, B. F. 2015. "BPI Challenge 2015," Eindhoven University of Technology. (https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1).
- **BPIC15_5 (BPI Challenge 2015, Municipality 5)**: van Dongen, B. F. 2015. "BPI Challenge 2015," Eindhoven University of Technology. (https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1).
- **BPIC17 (BPI Challenge 2017)**: van Dongen, Boudewijn (2017): BPI Challenge 2017. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
- **BPIC18 (BPI Challenge 2018, EU Agriculture Subsidy)**: van Dongen, Boudewijn; Borchert, F. (Florian) (2018): BPI Challenge 2018. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972
- **BPIC19 (BPI Challenge 2019)**: van Dongen, Boudewijn (2019): BPI Challenge 2019. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1
- **BPIC20_dd (BPI Challenge 2020, Domestic Declarations)**: van Dongen, Boudewijn (2020): BPI Challenge 2020: Domestic Declarations. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5
- **BPIC20_id (BPI Challenge 2020, International Declarations)**: van Dongen, Boudewijn (2020): BPI Challenge 2020: International Declarations. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5
- **BPIC20_ptc (BPI Challenge 2020, Prepaid Travel Costs)**: van Dongen, Boudewijn (2020): BPI Challenge 2020: Prepaid Travel Costs. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165
- **BPIC20_rfp (BPI Challenge 2020, Request For Payment)**: van Dongen, Boudewijn (2020): BPI Challenge 2020: Request For Payment. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030
- **BPIC20_tpd (BPI Challenge 2020, Travel Permit Data)**: van Dongen, Boudewijn (2020): BPI Challenge 2020: Travel Permit Data. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550
- **CALL**: Chapela-Campa, D., Benchekroun, I., Baron, O., Dumas, M., Krass, D., & Senderovich, A. (2024). Evaluation datasets and results of the paper "A Framework for Measuring the Quality of Business Process Simulation Models" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12126071
- **ITHD (Help Desk)**: Polato, Mirko (2017): Dataset belonging to the help desk log of an Italian Company. Version 1. 4TU.ResearchData. dataset. https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb
- **MOBIS (MobIS Challenge 2019)**: Scheid, Marting; Rehse, Jana-Rebecca; Houy, Constantin; Fettke, Peter (2018): Data Set for MobIS Challenge 2019. https://www.dfki.de/web/forschung/projekte-publikationen/publikation/10258
- **SEPSIS (Sepsis Cases)**: Mannhardt, F. 2016. Sepsis Cases - Event Log, Media types: application/x-gzip, text/csv, text/plain, text/xml, Eindhoven University of Technology, December 7. (https://doi.org/10.4121/UUID:915D2BFB-7E84-49AD-A286-DC35F063A460).
