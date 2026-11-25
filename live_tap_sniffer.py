"""Live network TAP sniffer with zero training-serving skew.

Fix highlights:
- Uses `LiveWindowExtractor` from `smart_city_ml` so `Payload_Entropy`, `IAT_Variance`,
  and protocol one-hot features match training 1:1.
- Fixes `MDNS` detection (UDP port 5353), plus MQTT (1883/8883), HTTP (80/8080/1880),
  and TLS (443) port mapping so `Rolling_MDNS_Count` is no longer permanently 0.
- Supports configurable `--count` (0 = continuous), `--iface`, and optional `--mqtt-alert`
  with cooldown rate-limiting.
"""

import argparse
import sys
import time
from typing import Dict, Optional

import joblib
import scapy.all as scapy

from smart_city_ml import (
    BASE_DIR,
    LIVE_CORE_FEATURES,
    LiveWindowExtractor,
    get_protocol_dummy_columns,
    send_mqtt_command,
)

MODEL_PATH = BASE_DIR / "production_model.pkl"
ALERT_TOWER_TOPIC = "city/alerts/tower"


def classify_packet_protocol(packet: scapy.Packet) -> str:
    """Map Scapy packet layers/ports onto the standardized training protocol categories."""
    if packet.haslayer(scapy.ARP):
        return "ARP"
    if packet.haslayer(scapy.UDP):
        sport = int(packet[scapy.UDP].sport)
        dport = int(packet[scapy.UDP].dport)
        if sport == 5353 or dport == 5353:
            return "MDNS"
        if sport == 53 or dport == 53 or packet.haslayer(scapy.DNS):
            return "DNS"
        if sport == 443 or dport == 443:
            return "QUIC"
        return "Other"
    if packet.haslayer(scapy.TCP):
        sport = int(packet[scapy.TCP].sport)
        dport = int(packet[scapy.TCP].dport)
        if sport in (1883, 8883) or dport in (1883, 8883):
            return "MQTT"
        if sport in (80, 8080, 1880) or dport in (80, 8080, 1880):
            return "HTTP"
        if sport == 443 or dport == 443:
            return "TLSv1.2"
        return "TCP"
    if packet.haslayer(scapy.DNS):
        return "DNS"
    return "Other"


def extract_ip_endpoints(packet: scapy.Packet) -> tuple[str, str]:
    """Return `(src, dst)` for IPv4 or Ethernet/ARP packets."""
    if packet.haslayer(scapy.IP):
        return str(packet[scapy.IP].src), str(packet[scapy.IP].dst)
    if packet.haslayer(scapy.ARP):
        return str(packet[scapy.ARP].psrc), str(packet[scapy.ARP].pdst)
    if packet.haslayer(scapy.Ether):
        return str(packet[scapy.Ether].src), str(packet[scapy.Ether].dst)
    return "Unknown", "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live ML Network TAP Sniffer")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to trained production_model.pkl")
    parser.add_argument("--count", type=int, default=500, help="Packets to capture (0 for unlimited)")
    parser.add_argument("--iface", default=None, help="Optional network interface name")
    parser.add_argument("--mqtt-alert", action="store_true", help="Publish MQTT alerts on detected anomalies")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Seconds between MQTT alerts per source IP")
    args = parser.parse_args()

    print("🦅 INITIALIZING LIVE ML NETWORK TAP SNIFFER 🦅")
    print("-" * 75)

    trained_model = None
    expected_cols = LIVE_CORE_FEATURES + get_protocol_dummy_columns()
    model_file = BASE_DIR / args.model if not args.model.startswith(str(BASE_DIR)) else args.model

    try:
        trained_model = joblib.load(model_file)
        if hasattr(trained_model, "feature_names_in_"):
            expected_cols = list(trained_model.feature_names_in_)
        print(f"🧠 Loaded pre-trained model ({len(expected_cols)} features).")
    except FileNotFoundError:
        print(f"⚠️ Model not found at '{model_file}'. Running in monitor-only mode.")
    except Exception as exc:
        print(f"⚠️ Failed to load model ({exc}). Running in monitor-only mode.")

    extractor = LiveWindowExtractor(max_buffer_size=50, expected_columns=expected_cols)
    last_alert_time: Dict[str, float] = {}

    def process_sniffed_packet(packet: scapy.Packet) -> None:
        try:
            if not (packet.haslayer(scapy.IP) or packet.haslayer(scapy.ARP)):
                return

            src_ip, dst_ip = extract_ip_endpoints(packet)
            proto_label = classify_packet_protocol(packet)
            feature_df = extractor.update(
                p_time=float(packet.time),
                p_length=float(len(packet)),
                proto_label=proto_label,
                src_ip=src_ip,
                dst_ip=dst_ip,
            )

            if trained_model is not None:
                prediction = int(trained_model.predict(feature_df)[0])
                status_flag = "🔥 [ANOMALY]" if prediction == -1 else "✅ [SECURE]"
                if prediction == -1 and args.mqtt_alert:
                    now = time.time()
                    if now - last_alert_time.get(src_ip, 0.0) >= args.cooldown:
                        ok, msg = send_mqtt_command(ALERT_TOWER_TOPIC, f"LIVE_ALERT_{src_ip}")
                        last_alert_time[src_ip] = now
                        print(f"   ⚡ MQTT Mitigation: {'OK' if ok else 'SKIP'} ({msg})")
            else:
                status_flag = "📡 [CAPTURE]"

            row = feature_df.iloc[0]
            print(
                f"{status_flag} {proto_label:<7} {src_ip:<15} -> {dst_ip:<15} | "
                f"Len={int(row['Length']):<5}B | mDNS={int(row['Rolling_MDNS_Count']):<2} | "
                f"IAT_Var={row['IAT_Variance']:.6f}"
            )
        except Exception as exc:
            print(f"⚠️ Packet processing error: {exc}", file=sys.stderr)

    print("🛡️ AI Defensive Core armed. Press Ctrl+C to stop.")
    print("-" * 75)
    try:
        scapy.sniff(prn=process_sniffed_packet, store=False, count=args.count, iface=args.iface)
    except KeyboardInterrupt:
        print("\n🛑 Live monitoring stopped by user.")
    print("-" * 75)


if __name__ == "__main__":
    main()
