# Stock price prediction (minute-level)
Steps to reproduce:

1. Setup environment:
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

2. DVC (optional): dvc init, dvc add data_versions (already tracked)

3. Run first iteration (use v0):
   dvc repro
   
   or
   
   python src/preprocess.py --data_dir data_versions/v0 --out_dir data/processed
   python src/featurize.py --in_dir data/processed --out_dir data/features
   python src/train.py --features_dir data/features --model_dir models --experiment_name iteration_v0
   python src/evaluate.py --features_dir data/features --model_dir models --out predictions/preds_v0.csv

5. Run second iteration (v0+v1):
  PYTHONPATH=. python src/preprocess.py --data_dir data_versions --out_dir data/processed_v1
  PYTHONPATH=. python src/featurize.py --in_dir data/processed_v1 --out_dir data/features_v1
  PYTHONPATH=. python src/train.py --features_dir data/features_v1 --model_dir models_v1 --experiment_name iteration_v1
  PYTHONPATH=. python src/evaluate.py --features_dir data/features_v1 --model_dir models_v1 --out predictions/preds_v1.csv


6. Tests: pytest
7. MLflow UI: mlflow ui --port 5000
