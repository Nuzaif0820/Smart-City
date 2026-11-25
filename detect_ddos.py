"""Volumetric DDoS and packet-rate anomaly scanner for Wireshark CSV logs."""

import argparse
from pathlib import Path

from smart_city_ml import BASE_DIR, load_wireshark_csv


def analyze_ddos(input_path: Path, pps_threshold: int = 5000) -> None:
    print(f"🛡️ Analyzing '{input_path.name}' for potential DDoS floods (threshold={pps_threshold} pps)...")
    df = load_wireshark_csv(input_path)

    print("\n🔝 Top 5 Traffic-Generating Devices:")
    print(df["Source"].value_counts().head(5).to_string())

    df["Second"] = df["Time"].fillna(0).astype(int)
    traffic_per_second = df["Second"].value_counts().sort_index()
    dangerous_seconds = traffic_per_second[traffic_per_second > pps_threshold]

    print(f"\n📈 Peak observed rate: {int(traffic_per_second.max()) if not traffic_per_second.empty else 0} packets/sec")
    print("🚨 DDoS Flood Detection Results:")
    if not dangerous_seconds.empty:
        print(f"⚠️ WARNING: {len(dangerous_seconds)} second(s) exceeded {pps_threshold} packets/sec.")
        print(dangerous_seconds.head(5).to_string())
    else:
        print("✅ Traffic flow rates look stable. No single-second floods above threshold.")

    print("\n🎯 Top 3 Targeted Destination Addresses:")
    print(df["Destination"].value_counts().head(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Wireshark CSV for volumetric DDoS bursts.")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_dataset_5.csv"))
    parser.add_argument("--threshold", type=int, default=5000)
    args = parser.parse_args()
    analyze_ddos(Path(args.input), pps_threshold=args.threshold)


if __name__ == "__main__":
    main()
