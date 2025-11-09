# feast_feature_repo/feature_repo.py
from feast import FeatureStore
from pathlib import Path
import os

def setup_and_materialize():
    # assumes this file executed from repo root
    repo_path = Path(__file__).parent
    os.chdir(repo_path)
    # You would normally run feast init / apply. For simplicity,
    # we will create a FeatureStore instance pointing to this directory,
    # which uses feature definitions placed here.
    fs = FeatureStore(repo_path=str(repo_path))
    # materialize -> for file source, we can skip online materialization in this minimal example
    print("Feast feature store available at", repo_path)
