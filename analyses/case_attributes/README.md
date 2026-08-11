# Case-attribute analysis

Three-step pipeline for inspecting case attributes and testing power-law
behaviour on selected attributes. Shared helpers live in `utils.rfc`
(`case_attribute_*` modules); these scripts are analysis-specific.

Outputs default to `results/case_attributes/` (per-log inventories/plots;
Clauset CSVs under `results/case_attributes/powerlaw/<model>/`).

Run from the repository root (so `data/` and `results/` resolve correctly):

```bash
# Full pipeline 01 → 02 → 03
python analyses/case_attributes/main.py --datasets TEST_BPIC12

# Subset
python analyses/case_attributes/main.py --datasets TEST_BPIC12 --from 02 --to 03
python analyses/case_attributes/main.py --datasets TEST_BPIC12 --only 01
```

Individual steps:

```bash
# 01) Inventory case attributes → editable attribute_inventory.csv
python analyses/case_attributes/01_describe.py --help

# 02) Attribute–variant relevance + selection CSV + plots
python analyses/case_attributes/02_analyze_relevance.py --help

# 03) Clauset-style power-law fits on selected attributes
python analyses/case_attributes/03_powerlaw_distributions.py --help
```

Order matters: step 02 expects inventory from step 01; step 03 expects selections
from step 02.
