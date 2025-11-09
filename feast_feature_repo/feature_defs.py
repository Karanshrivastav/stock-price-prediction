# feast_feature_repo/feature_defs.py
from feast import Entity, Feature, FeatureView, ValueType, FileSource
from feast import RepoConfig
from datetime import timedelta
from pathlib import Path

data_path = Path("../data/features/features.parquet").absolute()

# Source
fs = FileSource(
    path=str(data_path),
    timestamp_field="timestamp"
)

# Entity: symbol
symbol = Entity(name="symbol", value_type=ValueType.STRING, description="stock symbol")

fv = FeatureView(
    name="stock_minute_features",
    entities=["symbol"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="rolling_avg_10", dtype=ValueType.DOUBLE),
        Feature(name="volume_sum_10", dtype=ValueType.DOUBLE),
    ],
    online=True,
    input=fs,
    tags={}
)
