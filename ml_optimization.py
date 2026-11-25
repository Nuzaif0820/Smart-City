"""Advanced ML optimization comparison (Scaled Isolation Forest vs Scaled LOF)."""

import argparse
import time
from pathlib import Path

from smart_city_ml import (
    BASE_DIR,
    build_scaled_isolation_forest,
    build_scaled_lof,
    engineer_features,
    get_feature_columns,
    load_wireshark_csv,
)


def run_optimization(input_path: Path, output_path: Path, lof_sample_size: int = 50000) -> None:
    print(f"🔄 Loading raw network stream: {input_path.name}...")
    df = load_wireshark_csv(input_path)
    print(f"📊 Loaded {len(df):,} packet records.")

    df = engineer_features(df, include_protocol_dummies=True)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0.0)
    print(f"✅ Feature matrix shape: {X.shape}")

    t0 = time.time()
    iforest = build_scaled_isolation_forest(contamination=0.01, random_state=42)
    df["IForest_Anomaly"] = iforest.fit_predict(X)
    print(f"⏱️ Isolation Forest (scaled): {time.time() - t0:.2f}s")

    t0 = time.time()
    lof = build_scaled_lof(contamination=0.01, n_neighbors=20)
    sample_idx = X.head(lof_sample_size).index
    df.loc[sample_idx, "LOF_Anomaly"] = lof.fit_predict(X.loc[sample_idx])
    print(f"⏱️ Local Outlier Factor (scaled, {len(sample_idx):,} rows): {time.time() - t0:.2f}s")

    if_count = int((df["IForest_Anomaly"] == -1).sum())
    lof_count = int((df["LOF_Anomaly"] == -1).sum())
    overlap = int(((df["IForest_Anomaly"] == -1) & (df["LOF_Anomaly"] == -1)).sum())

    print("\n📊 --- CROSS-MODEL CORRELATION SUMMARY ---")
    print(f"├── Isolation Forest anomalies : {if_count:,}")
    print(f"├── LOF anomalies (sampled)    : {lof_count:,}")
    print(f"└── Joint consensus overlap    : {overlap:,}")

    df.to_csv(output_path, index=False)
    print(f"💾 Results saved to '{output_path.name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaled IForest vs LOF comparison")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_dataset_5.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "smart_city_advanced_ml_results.csv"))
    parser.add_argument("--lof-sample", type=int, default=50000)
    args = parser.parse_args()

    print("🤖 INITIALIZING SCALED ML OPTIMIZATION PIPELINE 🤖")
    print("-" * 60)
    run_optimization(Path(args.input), Path(args.output), lof_sample_size=args.lof_sample)
    print("-" * 60)


if __name__ == "__main__":
    main()
