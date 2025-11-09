# src/preprocess.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils import read_csv_files_from_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()

def ensure_minute_index(df):
    # Expect timestamp column as string convertible to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values(['source_file','timestamp']).reset_index(drop=True)
    out_frames = []
    for fname, g in df.groupby('source_file'):
        g = g.set_index('timestamp').sort_index()
        # create full minute index from min to max
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq='T') # minute freq
        g = g.reindex(full_idx)
        # forward fill missing minute rows using previous minute values
        g[['open','high','low','close','volume','source_file']] = g[['open','high','low','close','volume','source_file']].ffill()
        # some initial rows may be NaN if no previous value -> drop them (or backfill)
        g = g.fillna(method='bfill')
        g = g.reset_index().rename(columns={'index':'timestamp'})
        g['symbol'] = fname.split('__')[0]
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)

def main():
    args = parse_args()
    df = read_csv_files_from_dir(args.data_dir)
    df_pre = ensure_minute_index(df)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "processed_concat.parquet"
    df_pre.to_parquet(out_path, index=False)
    print("WROTE:", out_path)

if __name__=='__main__':
    main()
