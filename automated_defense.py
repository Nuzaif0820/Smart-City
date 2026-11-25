"""Automated Isolation Forest evaluation with rate-limited MQTT alerting."""

import argparse
import time
from pathlib import Path
from typing import Set

from smart_city_ml import (
    BASE_DIR,
    JETSON_SERVER_IP,
    build_scaled_isolation_forest,
    engineer_features,
    get_feature_columns,
    load_wireshark_csv,
    send_mqtt_command,
)

ALERT_TOWER_TOPIC = "city/alerts/tower"


def run_automated_defense(
    input_path: Path,
    broker_ip: str = JETSON_SERVER_IP,
    contamination: float = 0.01,
    max_alerts: int = 5,
    simulate_if_unreachable: bool = True,
) -> int:
    print(f"🔄 Loading traffic dataset: {input_path.name}...")
    df = load_wireshark_csv(input_path)
    df = engineer_features(df, include_protocol_dummies=True)
    features = get_feature_columns(df)
    X = df[features].fillna(0.0)

    print(f"🤖 Evaluating {len(X):,} packets using scaled Isolation Forest...")
    model = build_scaled_isolation_forest(contamination=contamination, random_state=42)
    df["Anomaly_Score"] = model.fit_predict(X)
    anomalies = df[df["Anomaly_Score"] == -1]

    print(f"📊 Flagged {len(anomalies):,} anomalies ({len(anomalies) / len(df) * 100:.2f}%).")

    sent_sources: Set[str] = set()
    alerts_sent = 0
    for _, row in anomalies.iterrows():
        src = str(row["Source"])
        if src in sent_sources:
            continue
        sent_sources.add(src)
        alerts_sent += 1

        print(
            f"\n🚨 [ANOMALY #{alerts_sent}] Packet {row['No.']} | {row['Protocol']} | "
            f"{row['Length']}B | {src} -> {row['Destination']}"
        )
        ok, msg = send_mqtt_command(ALERT_TOWER_TOPIC, f"ALERT_ON_PACKET_{row['No.']}", broker_ip=broker_ip)
        if ok:
            print(f"   ✅ {msg}")
        elif simulate_if_unreachable:
            print(f"   ℹ️ Simulation fallback: {msg}")
        else:
            print(f"   ❌ {msg}")
            break

        if alerts_sent >= max_alerts:
            break
        time.sleep(0.2)

    print("\n🔒 Defense routine evaluation completed.")
    return alerts_sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Isolation Forest + MQTT Alert Defense")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_dataset_5.csv"))
    parser.add_argument("--broker", default=JETSON_SERVER_IP)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--max-alerts", type=int, default=5)
    args = parser.parse_args()

    run_automated_defense(
        Path(args.input),
        broker_ip=args.broker,
        contamination=args.contamination,
        max_alerts=args.max_alerts,
    )


if __name__ == "__main__":
    main()
