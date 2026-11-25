"""Explainable AI (XAI) statistical attribution engine.

Fix highlights:
- Replaces the hardcoded conclusion string with dynamic Z-score ranking so the
  explanation accurately reflects whichever features actually deviated most.
- Supports diagnosing multiple verified anomalies via `--limit`.
"""

import argparse
from pathlib import Path

import pandas as pd

from smart_city_ml import BASE_DIR, NUMERIC_FEATURES, compute_zscore_attribution


def run_xai_diagnosis(benchmark_path: Path, threats_path: Path, limit: int = 3) -> None:
    if not benchmark_path.exists() or not threats_path.exists():
        raise FileNotFoundError("Missing benchmark or verified threat CSVs. Run benchmarking and validation first.")

    df = pd.read_csv(benchmark_path, encoding="latin1")
    threats = pd.read_csv(threats_path, encoding="latin1")

    if threats.empty:
        print("ℹ️ No consensus threats available to diagnose.")
        return

    features = [c for c in NUMERIC_FEATURES if c in df.columns]

    for idx, (_, target_packet) in enumerate(threats.head(limit).iterrows(), start=1):
        packet_no = target_packet.get("No.", "?")
        print(f"\n📋 [{idx}/{min(limit, len(threats))}] Diagnostic Attribution for Packet No. {packet_no}:")
        report, summary = compute_zscore_attribution(df, target_packet, features=features)

        print(f"{'Engineered Feature':<22} | {'Baseline Mean':<15} | {'Packet Value':<15} | {'Z-Score'}")
        print("-" * 75)
        for _, row in report.iterrows():
            print(
                f"{row['Feature']:<22} | {row['Baseline_Mean']:<15.4f} | "
                f"{row['Packet_Value']:<15.4f} | {row['Z_Score']:+.2f} σ"
            )
        print("-" * 75)
        print(f"🤖 DYNAMIC MODEL DIAGNOSIS:\n{summary}")
        print("-" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explainable AI Z-score anomaly attribution.")
    parser.add_argument("--benchmark", default=str(BASE_DIR / "smart_city_benchmark_results.csv"))
    parser.add_argument("--threats", default=str(BASE_DIR / "smart_city_verified_threats.csv"))
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    print("🔮 EXPLAINABLE AI (XAI) ATTRIBUTION ENGINE 🔮")
    print("-" * 75)
    run_xai_diagnosis(Path(args.benchmark), Path(args.threats), limit=args.limit)


if __name__ == "__main__":
    main()
