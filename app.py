"""Standalone CLI anomaly evaluation script (relative paths + scaled pipeline)."""

import argparse
from pathlib import Path

from smart_city_ml import (
    BASE_DIR,
    build_scaled_isolation_forest,
    engineer_features,
    get_feature_columns,
    load_wireshark_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Isolation Forest anomaly detection on a Wireshark CSV.")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_dataset_8.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "smart_city_anomalies_predicted_8.csv"))
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()

    input_path = Path(args.input)
    print(f"🔄 Step 1: Loading smart city dataset '{input_path.name}'...")
    df = load_wireshark_csv(input_path)

    print("🛠️ Step 2: Engineering behavioral and protocol features...")
    df = engineer_features(df, include_protocol_dummies=True)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0.0)

    print("🤖 Step 3: Running scaled Isolation Forest...")
    model = build_scaled_isolation_forest(contamination=args.contamination, random_state=42)
    df["Anomaly_Score"] = model.fit_predict(X)
    anomalies = df[df["Anomaly_Score"] == -1]

    print("\n📊 --- RESULTS ---")
    print(f"Total network packets analyzed: {len(df):,}")
    print(f"Number of unusual anomalies flagged: {len(anomalies):,}")
    print("\n👀 Previewing first 5 flagged anomalies:")
    print(anomalies[["No.", "Time", "Source", "Destination", "Protocol", "Length"]].head(5).to_string(index=False))

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Step 4: Results saved to '{output_path.name}'.")


if __name__ == "__main__":
    main()
