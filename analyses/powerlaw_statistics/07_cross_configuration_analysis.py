"""Step 07: cross-configuration sensitivity summary from saved Clauset results.

Writes ``configurations.csv`` and ``sensitivity_summary.csv`` under
``<experiment>/sensitivity``. Recalculates eligibility, GOF non-rejection,
paired comparisons, and A–E categories across the adaptive one-at-a-time
threshold settings. Fits, likelihood-ratio tests, and GOF bootstraps are not
rerun.

``main.py`` runs this step for real-log experiments after compose. It can
also be run alone:

    python analyses/powerlaw_statistics/07_cross_configuration_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.powerlaw_statistics.analysis_helpers.sensitivity import main as sensitivity_main


def main(argv: list[str] | None = None) -> None:
    sensitivity_main(argv)


if __name__ == "__main__":
    main()
