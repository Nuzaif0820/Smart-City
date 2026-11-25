"""Inter-model congruence and consensus threat extraction.

Fix highlights:
- Uses unique `Row_ID` (with fallback to `No.`) to avoid packet-number collisions across
  multi-file datasets.
- Reports pairwise Jaccard overlaps across all three models as well as 2-of-3 and 3-of-3
  consensus counts.
"""

import argparse
from pathlib import Path
from typing import Set

import pandas as pd

from smart_city_ml import BASE_DIR


def jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    union = len(set_a | set_b)
    return (len(set_a & set_b) / union) * 100.0 if union > 0 else 0.0


def run_congruence_validation(input_path: Path, output_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing '{input_path.name}'. Run model_benchmarking.py first.")

    df = pd.read_csv(input_path, encoding="latin1")
    id_col = "Row_ID" if "Row_ID" in df.columns else "No."

    iforest_ids = set(df[df["Anomaly_IForest"] == -1][id_col])
    lof_ids = set(df[df["Anomaly_LOF"] == -1][id_col])
    ocsvm_ids = set(df[df["Anomaly_OCSVM"] == -1][id_col])

    triple_consensus = iforest_ids & lof_ids & ocsvm_ids
    two_of_three = (iforest_ids & lof_ids) | (iforest_ids & ocsvm_ids) | (lof_ids & ocsvm_ids)

    print("📊 INTER-MODEL JACCARD OVERLAP CONGRUENCE REPORT:")
    print(f"├── Isolation Forest ∩ Local Outlier Factor : {jaccard_similarity(iforest_ids, lof_ids):.2f}% ({len(iforest_ids & lof_ids)} packets)")
    print(f"├── Isolation Forest ∩ One-Class SVM        : {jaccard_similarity(iforest_ids, ocsvm_ids):.2f}% ({len(iforest_ids & ocsvm_ids)} packets)")
    print(f"├── Local Outlier Factor ∩ One-Class SVM    : {jaccard_similarity(lof_ids, ocsvm_ids):.2f}% ({len(lof_ids & ocsvm_ids)} packets)")
    print(f"├── 2-of-3 Model Agreement                  : {len(two_of_three)} packets")
    print(f"└── 🎯 TRIPLE-CONSENSUS TARGETS IDENTIFIED  : {len(triple_consensus)} packets")
    print("-" * 75)

    verified = df[df[id_col].isin(triple_consensus)].copy()
    if verified.empty and two_of_three:
        print("ℹ️ No 3-of-3 consensus found; falling back to 2-of-3 high-confidence agreement.")
        verified = df[df[id_col].isin(two_of_three)].copy()

    if not verified.empty:
        preview_cols = [c for c in [id_col, "No.", "Source", "Destination", "Protocol", "Length", "IAT_Variance", "Rolling_Source_Burst"] if c in verified.columns]
        print("\n🚨 HIGH-CONFIDENCE VERIFIED ANOMALY PROFILES:")
        print(verified[preview_cols].head(10).to_string(index=False))
        verified.to_csv(output_path, index=False)
        print(f"\n💾 Verified anomalies saved to '{output_path.name}' ({len(verified)} rows).")
    else:
        verified.to_csv(output_path, index=False)
        print("ℹ️ No consensus overlaps found inside this partition.")

    print("-" * 75)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-model anomaly congruence.")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_benchmark_results.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "smart_city_verified_threats.csv"))
    args = parser.parse_args()

    print("📐 VALIDATION: INTER-MODEL CONGRUENCE MATRIX 📐")
    print("-" * 75)
    run_congruence_validation(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
