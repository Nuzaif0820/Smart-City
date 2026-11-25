"""Train and persist the production Isolation Forest pipeline.

Fix highlights:
- Uses the exact live-compatible feature matrix from `smart_city_ml` (including
  `P_*` protocol indicators, `Rolling_Source_Burst`, and `Is_Lab_Subnet`).
- Wraps `IsolationForest` in a `RobustScaler` `Pipeline` so feature scaling is saved
  inside `production_model.pkl` and applied automatically at inference time.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from smart_city_ml import (
    BASE_DIR,
    LIVE_CORE_FEATURES,
    build_scaled_isolation_forest,
    get_feature_columns,
)


def train_production_model(input_path: Path, model_path: Path, contamination: float = 0.01) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing '{input_path.name}'. Run feature_engineering.py first.")

    df = pd.read_csv(input_path, encoding="latin1")
    features = get_feature_columns(df, core_features=LIVE_CORE_FEATURES)
    X = df[features].fillna(0.0)

    print(f"🌲 Training production Isolation Forest pipeline on {len(X):,} rows x {len(features)} features...")
    pipeline = build_scaled_isolation_forest(contamination=contamination, random_state=42)
    pipeline.fit(X)

    joblib.dump(pipeline, model_path)
    print(f"💾 Production model pipeline saved to '{model_path.name}'!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save production anomaly detection pipeline.")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_enriched_features.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "production_model.pkl"))
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()

    train_production_model(Path(args.input), Path(args.output), contamination=args.contamination)


if __name__ == "__main__":
    main()
