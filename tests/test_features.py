# tests/test_features.py
import pandas as pd
import pathlib
def test_features_exist():
    p = pathlib.Path("data/features/features.parquet")
    assert p.exists()
    df = pd.read_parquet(p)
    assert 'rolling_avg_10' in df.columns
    assert 'volume_sum_10' in df.columns
    assert 'target' in df.columns

def test_target_binary():
    df = pd.read_parquet("data/features/features.parquet")
    vals = set(df['target'].dropna().unique())
    assert vals.issubset({0,1})

