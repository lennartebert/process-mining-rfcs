# RFC notebooks

Interactive and exploratory notebooks for rank-frequency-curve (RFC) analysis
and synthetic process simulation. These notebooks import `utils/` only (no local
analysis scripts in this folder).

A permalink copy of the interactive notebook also lives at the repository root
(`interactive_simulation_experiments.ipynb`).

```bash
jupyter notebook interactive_simulation_experiments.ipynb
jupyter notebook analyses/rfcs/analyze_powerlaw_single_log.ipynb
jupyter notebook analyses/rfcs/interactive_simulation_experiments.ipynb
jupyter notebook analyses/rfcs/preset_simulation_experiments.ipynb
```

Use the `process-mining-rfcs` Jupyter kernel.

Paths:

- Attachments: `results/attachments/<concept>/<log>/attachments.csv.gz`
- Simulation experiments: `results/rfcs/experiments/`
- Log info / RFC tables & plots: `results/rfcs/`

Related CLI pipeline:

- Power-law statistics / Clauset batch: [`analyses/powerlaw_statistics/`](../powerlaw_statistics/)
