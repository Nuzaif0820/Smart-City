"""Multi-algorithm benchmarking suite (Isolation Forest, LOF, One-Class SVM).

Fix highlights:
- Wraps all models in `RobustScaler` pipelines so packet `Length` does not overwhelm
  sub-millisecond `Time_Delta` / `IAT_Variance` in distance & RBF-kernel models.
- Uses the full feature matrix including preserved `P_*` protocol indicators.
- Uses stratified/uniform sampling across the enriched dataset rather than only `head(N)`
  when `sample_size < len(df)` (configurable via `--sample-size`).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from smart_city_ml import (
    BASE_DIR,
    build_scaled_isolation_forest,
    build_scaled_lof,
    build_scaled_ocsvm,
    get_feature_columns,
)


def run_benchmark(input_path: Path, output_path: Path, sample_size: int = 40000, contamination: float = 0.01) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing '{input_path.name}'. Run feature_engineering.py first.")

    df = pd.read_csv(input_path, encoding="latin1")
    if "Row_ID" not in df.columns:
        df.insert(0, "Row_ID", range(1, len(df) + 1))

    feature_space = get_feature_columns(df)
    X = df[feature_space].fillna(0.0)

    if sample_size and len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42).sort_values("Row_ID").copy()
        X_sample = X.loc[df_sample.index]
    else:
        df_sample = df.copy()
        X_sample = X

    print(f"🎯 Evaluating 3 scaled models on {len(df_sample):,} packets across {len(feature_space)} features.\n")

    t0 = time.time()
    iforest = build_scaled_isolation_forest(contamination=contamination, random_state=42)
    df_sample["Anomaly_IForest"] = iforest.fit_predict(X_sample)
    t_iforest = time.time() - t0

    t0 = time.time()
    lof = build_scaled_lof(contamination=contamination, n_neighbors=20)
    df_sample["Anomaly_LOF"] = lof.fit_predict(X_sample)
    t_lof = time.time() - t0

    t0 = time.time()
    ocsvm = build_scaled_ocsvm(nu=contamination)
    df_sample["Anomaly_OCSVM"] = ocsvm.fit_predict(X_sample)
    t_ocsvm = time.time() - t0

    print("📝 COMPILING SCALED METRIC ANALYSIS LOGS")
    print("-" * 75)
    print(f"{'Algorithm Archetype':<28} | {'Detected Anomalies':<20} | {'Rate':<8} | {'Speed'}")
    print("-" * 75)
    for name, col, elapsed in [
        ("Isolation Forest (Scaled)", "Anomaly_IForest", t_iforest),
        ("Local Outlier Factor (Scaled)", "Anomaly_LOF", t_lof),
        ("One-Class SVM (Scaled)", "Anomaly_OCSVM", t_ocsvm),
    ]:
        count = int(np.sum(df_sample[col] == -1))
        rate = (count / len(df_sample)) * 100.0
        print(f"{name:<28} | {count:<20} | {rate:5.2f}%   | {elapsed:.2f}s")
    print("-" * 75)

    df_sample.to_csv(output_path, index=False)
    print(f"💾 Benchmarking results saved to '{output_path.name}'.")
    return df_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark scaled anomaly detection algorithms.")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_enriched_features.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "smart_city_benchmark_results.csv"))
    parser.add_argument("--sample-size", type=int, default=40000)
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()

    print("🧪 MULTI-ALGORITHM BENCHMARKING SUITE 🧪")
    print("-" * 75)
    run_benchmark(Path(args.input), Path(args.output), args.sample_size, args.contamination)


if __name__ == "__main__":
    main()
