# src/evaluate.py
import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out", dest="out", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(Path(args.features_dir)/"features.parquet")
    df = df.dropna(subset=['target'])
    X = df[['rolling_avg_10','volume_sum_10']].fillna(0)
    y = df['target'].astype(int)
    model = joblib.load(Path(args.model_dir)/"rf_model.joblib")
    preds = model.predict(X)
    df_out = df[['timestamp','symbol']].copy()
    df_out['pred'] = preds
    df_out['target'] = y
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(outp, index=False)
    print("WROTE:", outp)
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds))

if __name__ == "__main__":
    main()
