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
conda activate rfcs-in-pm

# 2) run the default pipeline (log info + extract attachments + static RFC)
python main.py --datasets TEST_BPIC12 --concept variants
```

If you want to (re)calculate attachments from raw logs, place your `.xes` files under `data/`, create a copy of the data dictionary (to be placed `data/data_dictionary.json`) and ensure refer to the data sets in the dictionary.
If `results/.../attachments.csv.gz` already exists, extraction is skipped by default.

You can also run the simulation notebooks:

```bash
jupyter notebook interactive_simulation_experiments.ipynb
jupyter notebook preset_simulation_experiments.ipynb
```

## Repository structure

The repo is structured as follows:

- `main.py` is the top-level pipeline entrypoint.
- `interactive_simulation_experiments.ipynb` provides interactive simulation widgets.
- `preset_simulation_experiments.ipynb` runs the preconfigured paper-aligned experiment batches.
- `data/` contains event-log inputs and dataset metadata.
- `results/` contains generated outputs (attachments, RFC analysis artifacts, and experiment artifacts). Permanent, version-controlled results live under `results/perm/`.
- `scripts/` contains CLI entrypoints only.
- `utils/` contains reusable modules. `scripts/` may import from `utils/` but not the other way around.

## Activity definition

For attachment extraction, variant counting, and activity statistics (`log_info`), activities are derived from the event log after import:

- If the log defines an XES **Activity classifier**, each activity label combines all of that classifier’s keys (joined with `+`, e.g. `concept:name` and `lifecycle:transition` → `Register+start`).
- Otherwise, activities are taken from **`concept:name`** only.

The same logic lives in `utils/io/activity_labels.py` and is used by `extract_attachments` and `log_info`.

## Running the CLI Scripts

Main pipeline entrypoint:

```bash
python main.py --datasets TEST_BPIC12 --concept variants
```

Default `main.py` stages:

- `scripts/log_info.py`
- `scripts/extract_attachments.py` (skips extraction when `attachments.csv.gz` already exists)
- `scripts/rfc_powerlaw_analysis.py`

Optional stages are opt-in:

- `--run-pdf` enables `scripts/pdf_powerlaw_analysis.py`
- `--run-dynamic` enables `scripts/dynamic_rfc_analysis.py`
- `--run-alpha-correlation` enables `scripts/alpha_correlationy.py`

Examples:

```bash
# default pipeline
python main.py --datasets TEST_BPIC12 --concept variants

# include PDF stage
python main.py --datasets TEST_BPIC12 --concept variants --run-pdf

# include both optional stages
python main.py --datasets TEST_BPIC12 --concept variants --run-pdf --run-dynamic

# alpha correlation only (requires existing log_info.csv and rfc_static_analysis.csv)
python main.py --datasets TEST_BPIC12 --concept variants \
  --skip-log-info --skip-extract --skip-static --run-alpha-correlation
```

Individual script entrypoints:

```bash
python scripts/extract_attachments.py --help
python scripts/rfc_powerlaw_analysis.py --help
python scripts/pdf_powerlaw_analysis.py --help
python scripts/dynamic_rfc_analysis.py --help
python scripts/log_info.py --help
python scripts/alpha_correlationy.py --help
```

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
