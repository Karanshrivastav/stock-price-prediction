# src/featurize.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()

def featurize(df, rolling_window=10, target_horizon=5):
    # For each symbol, compute rolling_avg_10 (close t-9..t) and volume_sum_10 (t-9..t)
    features = []
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        g['rolling_avg_10'] = g['close'].rolling(window=rolling_window, min_periods=1).mean()
        g['volume_sum_10'] = g['volume'].rolling(window=rolling_window, min_periods=1).sum()
        # target: 1 if close at t+target_horizon > close at t else 0
        g['future_close'] = g['close'].shift(-target_horizon)
        g['target'] = (g['future_close'] > g['close']).astype(int)
        features.append(g)
    df_feat = pd.concat(features, ignore_index=True)
    return df_feat

def main():
    args = parse_args()
    in_path = Path(args.in_dir) / "processed_concat.parquet"
    df = pd.read_parquet(in_path)
    # read rolling_window and target_horizon from params.yaml by importing
    import yaml, os
    with open("params.yaml", "r") as fh:
        params = yaml.safe_load(fh)
    rolling_window = params.get('rolling_window', 10)
    target_horizon = params.get('target_horizon', 5)
    df_feat = featurize(df, rolling_window=rolling_window, target_horizon=target_horizon)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "features.parquet"
    df_feat.to_parquet(out_path, index=False)
    print("WROTE:", out_path)

if __name__=='__main__':
    main()
