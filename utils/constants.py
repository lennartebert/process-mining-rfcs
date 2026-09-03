from __future__ import annotations

from pathlib import Path

# Default location for the user-maintained dataset registry.
RESULTS_DIR = Path("results")
ATTACHMENTS_DIR = RESULTS_DIR / "attachments"
RFCS_DIR = RESULTS_DIR / "rfcs"
N_GRAMS_DIR = RESULTS_DIR / "n_grams"
N_GRAMS_REAL_DIR = N_GRAMS_DIR / "real"
N_GRAMS_SIM_DIR = N_GRAMS_DIR / "sim"
PREFERENTIAL_ATTACHMENT_DIR = RESULTS_DIR / "preferential_attachment"
CASE_ATTRIBUTES_DIR = RESULTS_DIR / "case_attributes"
# log_info.csv / .tex live directly under rfcs/ (not a nested log_info/ folder).
LOG_INFO_DIR = RFCS_DIR
TEST_RESULTS_DIR = RESULTS_DIR / "test"
TEST_ATTACHMENTS_DIR = TEST_RESULTS_DIR / "attachments"
TEST_N_GRAMS_DIR = TEST_RESULTS_DIR / "n_grams"
TEST_DATASET = "TEST_BPIC12"
# Manual copy+paste staging for git-tracked permanent results (never auto-written).
PERM_RESULTS_DIR = RESULTS_DIR / "perm"
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
ALL_SIM_LOG_DATASETS = [f"{name}_sim" for name in ALL_REAL_LOG_DATASETS]
ALL_SIM_LOGS_TOKEN = "ALL_SIM_LOGS"


