"""Autonomous closed-loop defense replay core.

Fix highlights:
- Fixes the false-positive condition where non-mDNS packets triggered mDNS alerts
  merely because background mDNS traffic existed in the trailing 30 packets.
- Replaces single-packet `break` termination with configurable rate-limited continuous
  monitoring (`--max-alerts`, `--cooldown`).
- Uses `smart_city_ml.send_mqtt_command` with clean simulation fallback.
"""

import argparse
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from smart_city_ml import BASE_DIR, JETSON_SERVER_IP, send_mqtt_command

ALERT_TOPIC = "city/alerts/tower"


def run_defense_loop(
    input_path: Path,
    broker_ip: str = JETSON_SERVER_IP,
    interval_sec: float = 0.005,
    max_alerts: int = 5,
    cooldown_sec: float = 2.0,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing '{input_path.name}'. Run model_benchmarking.py first.")

    df = pd.read_csv(input_path, encoding="latin1")
    print(f"📋 Streaming buffer mounted: {len(df):,} records from '{input_path.name}'.")
    print(f"🛡️ Monitoring for multi-model consensus anomalies (max_alerts={max_alerts})...")
    print("-" * 75)

    alerts_triggered = 0
    last_src_alert: Dict[str, float] = {}

    for packet in df.to_dict(orient="records"):
        votes = sum(
            int(packet.get(col) == -1)
            for col in ("Anomaly_IForest", "Anomaly_LOF", "Anomaly_OCSVM")
        )
        proto = str(packet.get("Protocol", "Unknown"))
        src = str(packet.get("Source", "Unknown"))
        dst = str(packet.get("Destination", "Unknown"))
        src_burst = float(packet.get("Rolling_Source_Burst", 0))
        mdns_count = float(packet.get("Rolling_MDNS_Count", 0))

        is_mdns_flood = proto == "MDNS" and mdns_count >= 5.0
        is_high_consensus = votes >= 3 or (votes >= 2 and (src_burst >= 15 or is_mdns_flood))

        if is_high_consensus:
            now = time.time()
            if now - last_src_alert.get(src, 0.0) >= cooldown_sec:
                alerts_triggered += 1
                last_src_alert[src] = now
                print(f"\n🔥 [CRITICAL ALARM #{alerts_triggered}] Votes={votes}/3")
                print(f"├── Packet No.: {packet.get('No.')} (Row_ID={packet.get('Row_ID', '-')})")
                print(f"├── Flow: {src} -> {dst} | Protocol={proto} | Length={packet.get('Length')}B")
                print(f"└── Burst={src_burst:.0f} | Rolling_MDNS={mdns_count:.0f}")

                ok, msg = send_mqtt_command(ALERT_TOPIC, f"EMERGENCY_ON_{src}", broker_ip=broker_ip)
                if ok:
                    print(f"✅ [ACTUATION] {msg}")
                else:
                    print(f"ℹ️ [SIMULATION] {msg}")
                print("-" * 75)

                if alerts_triggered >= max_alerts:
                    print(f"🛑 Reached configured alert cap ({max_alerts}). Stopping replay.")
                    break

        if interval_sec > 0:
            time.sleep(interval_sec)

    print(f"\n🏁 Streaming defense pass complete. Total alerts triggered: {alerts_triggered}")
    return alerts_triggered


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop autonomous defense replay.")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_benchmark_results.csv"))
    parser.add_argument("--broker", default=JETSON_SERVER_IP)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--max-alerts", type=int, default=5)
    args = parser.parse_args()

    print("🚨 AUTONOMOUS CLOSED-LOOP DEFENSE CORE 🚨")
    print("-" * 75)
    run_defense_loop(
        Path(args.input),
        broker_ip=args.broker,
        interval_sec=args.interval,
        max_alerts=args.max_alerts,
    )


if __name__ == "__main__":
    main()
