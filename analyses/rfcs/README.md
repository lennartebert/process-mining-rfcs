# RFC notebooks

Interactive and exploratory notebooks for rank-frequency-curve (RFC) analysis
and synthetic process simulation. These notebooks import `utils/` only (no local
analysis scripts in this folder).

```bash
jupyter notebook analyses/rfcs/analyze_powerlaw_single_log.ipynb
jupyter notebook analyses/rfcs/interactive_simulation_experiments.ipynb
jupyter notebook analyses/rfcs/preset_simulation_experiments.ipynb
```

Use the `process-mining-rfcs` Jupyter kernel.

Related CLI pipelines live elsewhere:

- N-grams / Clauset batch: [`analyses/n_grams/`](../n_grams/)
- Preferential attachment: [`analyses/preferential_attachment/`](../preferential_attachment/)
- Case attributes: [`analyses/case_attributes/`](../case_attributes/)
- Legacy helpers: [`analyses/others/`](../others/)
