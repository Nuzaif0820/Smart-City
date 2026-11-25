"""Feature engineering pipeline for Smart City network packet captures.

Fix highlights:
- Preserves one-hot `P_*` protocol columns when saving `smart_city_enriched_features.csv`.
- Defaults to lab-heavy datasets (where `172.16.16.*` OT/IoT traffic lives) instead of
  pure CDN background captures, while still supporting custom `--input` CSV files.
- Computes rolling windows per capture file before concatenating to prevent negative
  inter-file `Time_Delta` jumps, and adds a unique `Row_ID`.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from smart_city_ml import (
    BASE_DIR,
    engineer_features,
    get_feature_columns,
    load_wireshark_csv,
)

DEFAULT_LAB_DATASETS: List[str] = [
    "smart_city_dataset_5.csv",
    "smart_city_dataset_6.csv",
    "smart_city_dataset_7.csv",
    "smart_city_traffic_light.csv",
]


def run_feature_engineering(input_files: List[Path], output_path: Path, lab_only: bool = False) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in input_files:
        if not path.exists():
            print(f"⚠️ Skipping missing dataset: {path.name}")
            continue
        raw = load_wireshark_csv(path)
        if lab_only:
            raw = raw[
                raw["Source"].str.startswith("172.16.16.")
                | raw["Destination"].str.startswith("172.16.16.")
            ].copy()
        if raw.empty:
            continue
        raw["Source_File"] = path.name
        enriched = engineer_features(raw, include_protocol_dummies=True)
        frames.append(enriched)
        print(f"📊 Processed {path.name}: {len(enriched):,} packets")

    if not frames:
        raise FileNotFoundError("No valid input datasets found to engineer features.")

    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "Row_ID", range(1, len(combined) + 1))

    feature_cols = get_feature_columns(combined)
    print(f"✅ Engineering completed! Feature space: {len(feature_cols)} indicators ({len(combined):,} total rows).")
    combined.to_csv(output_path, index=False)
    print(f"💾 Enriched feature matrix saved to '{output_path.name}' (including all P_* protocol columns).")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart City Network Feature Engineering Pipeline")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Input Wireshark CSV file(s). Defaults to lab-heavy datasets (5, 6, 7, traffic_light).",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "smart_city_enriched_features.csv"),
        help="Output enriched CSV path.",
    )
    parser.add_argument(
        "--lab-only",
        action="store_true",
        help="Filter packets to those involving the 172.16.16.0/24 lab subnet.",
    )
    args = parser.parse_args()

    print("🛠️ INITIALIZING ADVANCED FEATURE ENGINEERING PIPELINE 🛠️")
    print("-" * 75)

    if args.input:
        files = [Path(p) if Path(p).is_absolute() else BASE_DIR / p for p in args.input]
    else:
        files = [BASE_DIR / name for name in DEFAULT_LAB_DATASETS]
        if not any(p.exists() for p in files):
            files = [BASE_DIR / "smart_city_dataset_2.csv"]

    run_feature_engineering(files, Path(args.output), lab_only=args.lab_only)
    print("-" * 75)


if __name__ == "__main__":
    main()
