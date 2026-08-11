# Preferential attachment

Scripts for measuring preferential attachment on attachment sequences.
Outputs default to `results/preferential_attachment/`.

```bash
python analyses/preferential_attachment/detect_preferential_attachment.py --help
```

Example (after attachments exist under `results/attachments/<concept>/<dataset>/`):

```bash
python analyses/preferential_attachment/detect_preferential_attachment.py \
  --inputs TEST_BPIC12=results/attachments/variants/TEST_BPIC12/attachments.csv.gz
```
