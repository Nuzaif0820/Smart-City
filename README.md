# Smart City CPS Anomaly Detection & Autonomous Defense

An end-to-end Cyber-Physical Systems (CPS) network telemetry, multi-model anomaly detection, Explainable AI (XAI), and MQTT closed-loop actuation platform built for a Smart City OT/IoT testbed (`172.16.16.0/24`, Nvidia Jetson MQTT broker at `172.16.16.100:1883`).

## Key Capabilities

- **Unified Feature & Scaling Pipeline (`smart_city_ml.py`)**: Shared vectorized batch feature extraction (`engineer_features`) and sliding-window live extraction (`LiveWindowExtractor`) with `RobustScaler` pipelines to eliminate training–serving skew.
- **Multi-Algorithm Benchmarking (`model_benchmarking.py`, `ml_optimization.py`)**: Compares scaled Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM models across packet timing (`Time_Delta`, `IAT_Variance`), structural proxies, protocol bursts (`Rolling_ARP_Count`, `Rolling_MDNS_Count`, `Rolling_Source_Burst`), lab subnet indicators (`Is_Lab_Subnet`), and one-hot `P_*` protocol encodings.
- **Inter-Model Congruence & Dynamic XAI (`validation_congruence.py`, `explainable_ai.py`)**: Computes Jaccard consensus overlaps and ranks actual per-feature Z-score deviations for flagged packets.
- **Live TAP Sniffer & Closed-Loop MQTT Defense (`live_tap_sniffer.py`, `live_defense_core.py`, `automated_defense.py`)**: Live Scapy packet sniffing (with UDP `5353` mDNS and TCP `1883/8883` MQTT classification) and rate-limited MQTT mitigation commands with safe simulation fallback.
- **Interactive Dashboards (`dashboard_v3.py`, `dashboard_v2.py`, `dashboard.py`)**: Streamlit UIs for ingestion, volumetric DDoS PPS monitoring, consensus inspection, node ping checks, and explicit MQTT actuation.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Engineer features from lab captures (or pass --input smart_city_traffic_light.csv)
python feature_engineering.py --input smart_city_traffic_light.csv

# 2. Benchmark scaled Isolation Forest, LOF, and One-Class SVM
python model_benchmarking.py

# 3. Validate consensus overlaps and inspect dynamic XAI attribution
python validation_congruence.py
python explainable_ai.py

# 4. Train and save production_model.pkl
python train_and_save_model.py

# 5. Launch the flagship Streamlit dashboard
streamlit run dashboard_v3.py
```

## Running Tests

```bash
pytest -v
```
