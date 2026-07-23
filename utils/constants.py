from __future__ import annotations

from pathlib import Path

# Default location for the user-maintained dataset registry.
RESULTS_DIR = Path("results")
LOG_INFO_DIR = RESULTS_DIR / "log_info"
TEST_RESULTS_DIR = RESULTS_DIR / "test"
TEST_DATASET = "TEST_BPIC12"
DATA_DIR = Path("data")
DATA_DICTIONARY_FILENAME = "data_dictionary.json"
DATA_DICTIONARY_PATH = DATA_DIR / DATA_DICTIONARY_FILENAME

# Shortcut for the standard real-log benchmark set.
ALL_REAL_LOG_DATASETS = [
    "ACCRE",
    "BPIC11",
    "BPIC12",
    "BPIC13_cp",
    "BPIC13_i",
    "BPIC14",
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
    "BPIC17",
    "BPIC18",
    "BPIC19",
    "BPIC20_dd",
    "BPIC20_id",
    "BPIC20_ptc",
    "BPIC20_rfp",
    "BPIC20_tpd",
    "CALL",
    "ITHD",
    "MOBIS",
    "RTFMP",
    "SEPSIS",
]
ALL_REAL_LOGS_TOKEN = "ALL_REAL_LOGS"


