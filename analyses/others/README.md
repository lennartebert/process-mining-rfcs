# Other / legacy analysis scripts

Unnumbered helpers that are not part of the main `powerlaw_statistics`,
`preferential_attachment`, or `case_attributes` pipelines. Candidates for later
removal.

Defaults write under `results/rfcs/` (tables under `rfcs/<analysis-name>/`,
per-log plot PDFs under `rfcs/<log>/`). Attachments are read from
`results/attachments/`.

| Script | Role |
|--------|------|
| `fit_static.py` | Static RFC fits/plots from attachments |
| `fit_pdf.py` | PDF power-law fits from attachments |
| `alpha_correlation.py` | Correlate log-size metrics with fitted alpha |
| `integrity_checks.py` | Compare `log_info.csv` to attachment extracts |

```bash
python analyses/others/fit_static.py --help
python analyses/others/fit_pdf.py --help
python analyses/others/alpha_correlation.py --help
python analyses/others/integrity_checks.py --help
```
