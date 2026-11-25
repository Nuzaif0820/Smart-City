"""Shared feature engineering, model helpers, XAI attribution, and MQTT utilities
for the Smart City CPS Anomaly Detection & Defense platform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

BASE_DIR = Path(__file__).resolve().parent
JETSON_SERVER_IP = "172.16.16.100"
MQTT_PORT = 1883
LAB_SUBNET_PREFIX = "172.16.16."

TOP_PROTOCOLS: List[str] = [
    "TCP",
    "HTTP",
    "TLSv1.2",
    "TLSv1.3",
    "DNS",
    "ARP",
    "MDNS",
    "QUIC",
    "MQTT",
    "SSLv2",
]

NUMERIC_FEATURES: List[str] = [
    "Length",
    "Time_Delta",
    "IAT_Variance",
    "Rolling_Mean_Length",
    "Rolling_Std_Length",
    "Payload_Entropy",
    "Rolling_ARP_Count",
    "Rolling_MDNS_Count",
    "Rolling_Source_Burst",
    "Is_Lab_Subnet",
]

# Live-sniffer compatible features (exact parity between offline training and live capture)
LIVE_CORE_FEATURES: List[str] = [
    "Length",
    "Time_Delta",
    "IAT_Variance",
    "Payload_Entropy",
    "Rolling_MDNS_Count",
    "Rolling_ARP_Count",
    "Rolling_Source_Burst",
    "Is_Lab_Subnet",
]

DEVICES: Dict[str, str] = {
    "Nvidia Jetson (Server/DB)": "172.16.16.100",
    "Tollgate 1 (Arduino)": "172.16.16.101",
    "Tollgate 2 (Arduino)": "172.16.16.102",
    "Tollgate 1 Display (ESP8266)": "172.16.16.103",
    "Tollgate 2 Display (ESP8266)": "172.16.16.104",
    "Airport Baggage/Flight Sensor (Pi)": "172.16.16.105",
    "Villa 2 Motion Sensor/Relays (Pi)": "172.16.16.106",
    "Villa 2 Fire/Light Sensor (Pi)": "172.16.16.107",
    "Gas / Hotel Fire Sensors (Pi)": "172.16.16.108",
    "Traffic Lights Controller (Pi)": "172.16.16.109",
    "Stock Exchange Billboard (Pi)": "172.16.16.110",
    "Stock Exchange Display (Pi)": "172.16.16.111",
    "Air Quality/Hospital & Ind Fire (Pi)": "172.16.16.112",
    "Smart Parking Gate/HVAC Fire (Pi)": "172.16.16.113",
    "Smart Parking Sensors/Airport BB (Pi)": "172.16.16.114",
    "Check-in Sensor/Arrival Display (Pi)": "172.16.16.115",
    "Boarding Pass KIOSK (Pi)": "172.16.16.116",
    "Central Fire Station (Pi)": "172.16.16.117",
    "Smartmeter 1 (ESP8266)": "172.16.16.118",
    "Smartmeter 2 (ESP8266)": "172.16.16.119",
    "Hospital Patients Records (ESP8266)": "172.16.16.120",
    "Power Station Turbine (Siemens PLC)": "172.16.16.121",
    "Water Treatment Plant (Siemens PLC)": "172.16.16.122",
    "Airport Entry Gate (Siemens PLC)": "172.16.16.123",
    "Airport Arrival/Departure Gate (PLC)": "172.16.16.124",
    "Airport Exit Gate / ATC Tower (PLC)": "172.16.16.125",
    "Data Center HVAC (Siemens PLC)": "172.16.16.126",
    "Metro Track Switching (Siemens PLC)": "172.16.16.127",
    "Check-in Conveyor Belt (Siemens PLC)": "172.16.16.128",
    "Lift Mechanism (Siemens PLC)": "172.16.16.129",
    "Baggage Collection Belt (Siemens PLC)": "172.16.16.130",
    "Metal Detector Gate (Siemens PLC)": "172.16.16.131",
    "Alert Towers (Siemens PLC)": "172.16.16.132",
    "TDS Sensor (ESP8266)": "172.16.16.133",
    "IP Camera 1 (Airport)": "172.16.16.134",
    "IP Camera 2 (Stock Exchange)": "172.16.16.135",
    "Alexa Integration (Pi)": "172.16.16.136",
    "Table Router Gateway": "172.16.16.1",
}


def compute_structural_entropy(length: float, time_delta: float) -> float:
    """Compute structural length-timing proxy consistently across batch and live paths."""
    safe_len = max(float(length), 1.0)
    safe_dt = max(float(time_delta), 0.0)
    return float(np.log2(safe_len) * np.log2(safe_dt + 1.0)) if safe_dt > 0.0 else 0.0


def load_wireshark_csv(source: Any) -> pd.DataFrame:
    """Load a Wireshark CSV export with standard fallback encoding and column validation."""
    df = pd.read_csv(source, encoding="latin1")
    required = {"Time", "Length", "Protocol", "No.", "Source", "Destination"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required Wireshark columns: {missing}")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce").fillna(0.0)
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(0.0)
    df["Protocol"] = df["Protocol"].fillna("Other").astype(str)
    df["Source"] = df["Source"].fillna("Unknown").astype(str)
    df["Destination"] = df["Destination"].fillna("Unknown").astype(str)
    return df


def compute_rolling_source_burst(sources: Iterable[str], window: int = 30) -> np.ndarray:
    """Count occurrences of the current Source IP within the trailing `window` packets."""
    from collections import Counter, deque

    history: deque[str] = deque()
    counts: Counter[str] = Counter()
    result: List[int] = []
    for src in sources:
        history.append(src)
        counts[src] += 1
        if len(history) > window:
            evicted = history.popleft()
            counts[evicted] -= 1
            if counts[evicted] <= 0:
                del counts[evicted]
        result.append(counts[src])
    return np.asarray(result, dtype=np.int32)


def engineer_features(df: pd.DataFrame, include_protocol_dummies: bool = True) -> pd.DataFrame:
    """Compute vectorized behavioral and protocol features without dropping encoded columns."""
    out = df.copy()
    out["Time_Delta"] = out["Time"].diff().fillna(0.0).clip(lower=0.0)
    out["IAT_Variance"] = out["Time_Delta"].rolling(window=10, min_periods=1).var().fillna(0.0)
    out["Rolling_Mean_Length"] = out["Length"].rolling(window=10, min_periods=1).mean().fillna(0.0)
    out["Rolling_Std_Length"] = out["Length"].rolling(window=10, min_periods=1).std().fillna(0.0)

    out["Is_ARP"] = (out["Protocol"] == "ARP").astype(int)
    out["Is_MDNS"] = (out["Protocol"] == "MDNS").astype(int)
    out["Rolling_ARP_Count"] = out["Is_ARP"].rolling(window=30, min_periods=1).sum()
    out["Rolling_MDNS_Count"] = out["Is_MDNS"].rolling(window=30, min_periods=1).sum()

    out["Length_Log"] = np.log2(out["Length"].clip(lower=1.0))
    out["Payload_Entropy"] = out["Length_Log"] * np.log2(out["Time_Delta"] + 1.0)

    out["Is_Lab_Subnet"] = (
        out["Source"].astype(str).str.startswith(LAB_SUBNET_PREFIX)
        | out["Destination"].astype(str).str.startswith(LAB_SUBNET_PREFIX)
    ).astype(int)
    out["Rolling_Source_Burst"] = compute_rolling_source_burst(out["Source"].astype(str), window=30)

    if include_protocol_dummies:
        out["Proto_Group"] = out["Protocol"].where(out["Protocol"].isin(TOP_PROTOCOLS), "Other")
        for proto in [*TOP_PROTOCOLS, "Other"]:
            out[f"P_{proto}"] = (out["Proto_Group"] == proto).astype(int)

    return out


def get_protocol_dummy_columns() -> List[str]:
    """Return deterministic ordered list of one-hot protocol indicator columns."""
    return [f"P_{proto}" for proto in [*TOP_PROTOCOLS, "Other"]]


def get_feature_columns(df: pd.DataFrame, core_features: Optional[List[str]] = None) -> List[str]:
    """Return ordered feature columns present in `df`, combining core and `P_*` indicators."""
    base = list(core_features or NUMERIC_FEATURES)
    proto_cols = [c for c in get_protocol_dummy_columns() if c in df.columns]
    return [c for c in base if c in df.columns] + proto_cols


def build_scaled_isolation_forest(contamination: float = 0.01, random_state: int = 42) -> Pipeline:
    """Create a scaled Isolation Forest pipeline."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)),
    ])


def build_scaled_lof(contamination: float = 0.01, n_neighbors: int = 20) -> Pipeline:
    """Create a scaled Local Outlier Factor pipeline."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)),
    ])


def build_scaled_ocsvm(nu: float = 0.01) -> Pipeline:
    """Create a scaled One-Class SVM pipeline so features are normalized before RBF kernel."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", OneClassSVM(nu=nu, kernel="rbf", gamma="scale")),
    ])


def compute_zscore_attribution(
    df: pd.DataFrame,
    target_row: pd.Series,
    features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Compute per-feature Z-scores and generate a data-driven diagnosis string."""
    cols = [c for c in (features or NUMERIC_FEATURES) if c in df.columns and c in target_row.index]
    means = df[cols].mean()
    stds = df[cols].std().replace(0, 1.0).fillna(1.0)

    records = []
    for col in cols:
        val = float(target_row[col])
        mean_val = float(means[col])
        std_val = float(stds[col]) if stds[col] != 0 else 1.0
        z = (val - mean_val) / std_val
        records.append({
            "Feature": col,
            "Baseline_Mean": mean_val,
            "Packet_Value": val,
            "Z_Score": z,
            "Abs_Z": abs(z),
        })

    report = pd.DataFrame(records).sort_values("Abs_Z", ascending=False).reset_index(drop=True)
    top_drivers = report.head(2)
    if top_drivers.empty:
        summary = "Insufficient numerical features to generate attribution."
    else:
        driver_parts = [
            f"**{r['Feature']}** ({r['Z_Score']:+.2f}σ vs baseline {r['Baseline_Mean']:.3f})"
            for _, r in top_drivers.iterrows()
        ]
        summary = (
            f"Flagged packet `{target_row.get('No.', '?')}` ({target_row.get('Protocol', 'Unknown')} "
            f"from `{target_row.get('Source', '?')}` to `{target_row.get('Destination', '?')}`) "
            f"driven primarily by {' and '.join(driver_parts)}."
        )
    return report, summary


def send_mqtt_command(
    topic: str,
    payload: str,
    broker_ip: str = JETSON_SERVER_IP,
    port: int = MQTT_PORT,
    timeout: int = 5,
) -> Tuple[bool, str]:
    """Publish an MQTT actuation message and return `(success, message)`."""
    try:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        client.connect(broker_ip, port, timeout)
        info = client.publish(topic, payload)
        info.wait_for_publish(timeout=timeout)
        client.disconnect()
        return True, f"Published `{payload}` to `{topic}` on `{broker_ip}:{port}`."
    except Exception as exc:
        return False, f"Broker `{broker_ip}:{port}` unreachable: {exc}"


class LiveWindowExtractor:
    """Sliding-window feature extractor matching `engineer_features` exactly for live sniffing."""

    def __init__(self, max_buffer_size: int = 50, expected_columns: Optional[List[str]] = None):
        from collections import deque

        self.max_buffer_size = max_buffer_size
        self.expected_columns = expected_columns
        self.buffer: deque[Dict[str, Any]] = deque(maxlen=max_buffer_size)

    def update(self, p_time: float, p_length: float, proto_label: str, src_ip: str, dst_ip: str) -> pd.DataFrame:
        self.buffer.append({
            "Time": float(p_time),
            "Length": float(p_length),
            "Protocol": str(proto_label),
            "Source": str(src_ip),
            "Destination": str(dst_ip),
        })

        times = np.array([row["Time"] for row in self.buffer], dtype=np.float64)
        lengths = np.array([row["Length"] for row in self.buffer], dtype=np.float64)
        protos = [row["Protocol"] for row in self.buffer]
        sources = [row["Source"] for row in self.buffer]

        if len(times) >= 2:
            deltas = np.maximum(np.diff(times), 0.0)
            deltas = np.insert(deltas, 0, 0.0)
            current_dt = float(deltas[-1])
            iat_window = deltas[-10:]
            iat_var = float(np.var(iat_window, ddof=1)) if len(iat_window) >= 2 else 0.0
        else:
            current_dt = 0.0
            iat_var = 0.0

        len_window = lengths[-10:]
        rolling_mean_len = float(np.mean(len_window))
        rolling_std_len = float(np.std(len_window, ddof=1)) if len(len_window) >= 2 else 0.0

        recent_30_protos = protos[-30:]
        rolling_arp = int(sum(1 for p in recent_30_protos if p == "ARP"))
        rolling_mdns = int(sum(1 for p in recent_30_protos if p == "MDNS"))
        recent_30_sources = sources[-30:]
        rolling_src_burst = int(sum(1 for s in recent_30_sources if s == src_ip))

        is_lab = int(
            src_ip.startswith(LAB_SUBNET_PREFIX) or dst_ip.startswith(LAB_SUBNET_PREFIX)
        )
        entropy = compute_structural_entropy(p_length, current_dt)

        row_dict: Dict[str, Any] = {
            "Length": float(p_length),
            "Time_Delta": current_dt,
            "IAT_Variance": iat_var if not np.isnan(iat_var) else 0.0,
            "Rolling_Mean_Length": rolling_mean_len,
            "Rolling_Std_Length": rolling_std_len if not np.isnan(rolling_std_len) else 0.0,
            "Payload_Entropy": entropy,
            "Rolling_ARP_Count": rolling_arp,
            "Rolling_MDNS_Count": rolling_mdns,
            "Rolling_Source_Burst": rolling_src_burst,
            "Is_Lab_Subnet": is_lab,
        }

        mapped_proto = proto_label if proto_label in TOP_PROTOCOLS else "Other"
        for proto in [*TOP_PROTOCOLS, "Other"]:
            row_dict[f"P_{proto}"] = int(mapped_proto == proto)

        cols = self.expected_columns or get_feature_columns(pd.DataFrame([row_dict]))
        return pd.DataFrame([{c: row_dict.get(c, 0) for c in cols}], columns=cols)
