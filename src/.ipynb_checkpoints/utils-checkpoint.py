# src/utils.py
import pandas as pd
import numpy as np
from pathlib import Path

def read_csv_files_from_dir(data_dir: str):
    p = Path(data_dir)
    files = sorted([f for f in p.glob("*.csv")])
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # ensure uniform timestamp col name
        if 'timestamp' not in df.columns:
            # try lower-case names
            df.columns = [c.lower() for c in df.columns]
        df['source_file'] = f.name
        dfs.append(df)
    if len(dfs)==0:
        raise RuntimeError(f"No CSV files found in {data_dir}")
    return pd.concat(dfs, ignore_index=True)
