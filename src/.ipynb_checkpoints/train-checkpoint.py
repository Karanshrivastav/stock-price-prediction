# src/train.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--experiment_name", default="stock_pred")
    return p.parse_args()

def main():
    args = parse_args()
    with open("params.yaml", "r") as fh:
        params = yaml.safe_load(fh)
    model_params = params.get('model', {})
    random_seed = params.get('random_seed', 42)
    tuning = params.get('tuning', {})

    df = pd.read_parquet(Path(args.features_dir)/"features.parquet")
    # drop rows where target is NaN (end of series)
    df = df.dropna(subset=['target'])
    # features
    X = df[['rolling_avg_10','volume_sum_10']].fillna(0)
    y = df['target'].astype(int)

    # time-based split: we will sort by timestamp and split by index to avoid leakage
    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int((1 - params.get('test_size', 0.2)) * len(df))
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        # basic model
        clf = RandomForestClassifier(**model_params)
        # optional hyperparam tuning
        param_grid = tuning.get('param_grid', {})
        n_iter = tuning.get('n_iter', 0)
        if n_iter and param_grid:
            rs = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=n_iter, cv=3, random_state=random_seed, n_jobs=-1)
            rs.fit(X_train, y_train)
            best = rs.best_estimator_
            mlflow.log_param("tuning", True)
            mlflow.log_params(rs.best_params_)
            model = best
        else:
            model = clf.fit(X_train, y_train)
            mlflow.log_param("tuning", False)
            mlflow.log_params(model_params)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        except Exception:
            auc = None

        mlflow.log_metric("accuracy", float(acc))
        if auc is not None:
            mlflow.log_metric("auc", float(auc))

        # save model
        md = Path(args.model_dir)
        md.mkdir(parents=True, exist_ok=True)
        model_path = md / "rf_model.joblib"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "sk_model")
        print("Saved model to", model_path)

if __name__ == "__main__":
    main()
