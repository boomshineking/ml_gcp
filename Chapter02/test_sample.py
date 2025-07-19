# tests/test_raw_data.py
from pathlib import Path
import pandas as pd
import pytest

# Resolve the repo root (MLGCP)
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
TRAIN_CSV = ARTIFACTS / "train.csv"
TEST_CSV  = ARTIFACTS / "test.csv"

def test_csv_files_exist():
    assert TRAIN_CSV.is_file(), f"Missing: {TRAIN_CSV}"
    assert TEST_CSV.is_file(),  f"Missing: {TEST_CSV}"

def test_nonzero_rows():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    assert df_train.shape[0] > 0, "train.csv is empty"
    assert df_test.shape[0]  > 0, "test.csv is empty"

def test_column_consistency():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    assert list(df_train.columns) == list(df_test.columns), (
        "train/test columns differ:\n"
        f" train: {df_train.columns.tolist()}\n"
        f" test:  {df_test.columns.tolist()}"
    )