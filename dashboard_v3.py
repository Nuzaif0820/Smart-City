"""Smart City Autonomous AI Defense Hub (v3 Flagship — Unified & Production-Aligned)."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import streamlit as st

from health_check import ping_host
from smart_city_ml import (
    BASE_DIR,
    DEVICES,
    JETSON_SERVER_IP,
    build_scaled_isolation_forest,
    build_scaled_lof,
    build_scaled_ocsvm,
    compute_zscore_attribution,
    engineer_features,
    get_feature_columns,
    load_wireshark_csv,
    send_mqtt_command,
)

st.set_page_config(page_title="Smart City AI Defense Hub", page_icon="🏙️", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0f172a; }
    h1, h2, h3 { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px 0 rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-card h3 { color: #94a3b8 !important; font-size: 0.95rem; margin-bottom: 6px; }
    .metric-card h2 { color: #38bdf8 !important; font-size: 2rem; font-weight: bold; margin: 0; }
    .metric-card p { color: #64748b; font-size: 0.82rem; margin-top: 4px; }
    .threat-banner {
        background: linear-gradient(90deg, #451a03 0%, #78350f 100%);
        border-left: 8px solid #f59e0b;
        padding: 20px;
        border-radius: 12px;
        color: #fef3c7;
        margin-bottom: 20px;
    }
    .stButton>button {
        border-radius: 10px;
        height: 3.4em;
        font-weight: bold;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
st.sidebar.title("🏙️ System Sentinel Core")
st.sidebar.caption("Unified ML Anomaly Detection, XAI Attribution, and CPS MQTT Control.")
st.sidebar.markdown("---")

nav = st.sidebar.radio(
    "Workspace View",
    ["🛡️ AI Defense Pipeline", "🖥️ Lab Node Scanner & MQTT Controls"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ Host Telemetry")
cpu = psutil.cpu_percent()
ram = psutil.virtual_memory().percent
disk = psutil.disk_usage(os.path.abspath(os.sep)).percent
st.sidebar.write(f"**CPU:** {cpu}% | **RAM:** {ram}% | **Disk:** {disk}%")
st.sidebar.progress(min(cpu / 100.0, 1.0))

broker_ip = st.sidebar.text_input("MQTT Broker IP", value=JETSON_SERVER_IP)
live_mqtt = st.sidebar.checkbox("Auto-send MQTT Alerts on Consensus", value=False)

st.title("🏙️ Smart City Autonomous AI Defense Center")

if nav == "🛡️ AI Defense Pipeline":
    st.markdown("Scaled multi-algorithm anomaly detection with dynamic Z-score XAI and volumetric DDoS checks.")
    st.markdown("---")

    col_src1, col_src2 = st.columns([2, 1])
    with col_src1:
        uploaded_file = st.file_uploader("Upload Wireshark CSV capture:", type=["csv"])
    with col_src2:
        local_csvs = sorted(p.name for p in BASE_DIR.glob("smart_city_*.csv") if p.stat().st_size < 50_000_000)
        default_idx = local_csvs.index("smart_city_traffic_light.csv") if "smart_city_traffic_light.csv" in local_csvs else 0
        selected_local = st.selectbox("Or load a local workspace CSV:", ["(None)"] + local_csvs, index=(default_idx + 1 if local_csvs else 0))

    raw_df = None
    dataset_label = None
    if uploaded_file is not None:
        raw_df = load_wireshark_csv(uploaded_file)
        dataset_label = uploaded_file.name
    elif selected_local != "(None)":
        local_path = Path(BASE_DIR / selected_local)
        if local_path.exists():
            raw_df = load_wireshark_csv(local_path)
            dataset_label = selected_local

    if raw_df is not None:
        lab_ratio = (
            raw_df["Source"].str.startswith("172.16.16.") | raw_df["Destination"].str.startswith("172.16.16.")
        ).mean() * 100.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><h3>📊 Ingested Packets</h3><h2>{len(raw_df):,}</h2><p>{dataset_label}</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h3>🌐 Protocols</h3><h2>{raw_df['Protocol'].nunique()}</h2><p>Distinct Layers</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><h3>🖥️ Source Nodes</h3><h2>{raw_df['Source'].nunique()}</h2><p>Unique Senders</p></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card'><h3>🏭 Lab Subnet Share</h3><h2>{lab_ratio:.1f}%</h2><p>172.16.16.0/24 Traffic</p></div>", unsafe_allow_html=True)

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        sample_limit = col_opt1.slider("Evaluation Window Size", 1000, 40000, min(15000, len(raw_df)), step=1000)
        contamination = col_opt2.slider("Target Contamination Rate", 0.005, 0.05, 0.01, step=0.005)
        include_ocsvm = col_opt3.checkbox("Include Scaled One-Class SVM", value=True)

        if st.button("🚀 RUN SCALED FEATURE & MULTI-MODEL ANALYSIS", type="primary"):
            with st.status("Running unified feature & scaled ML pipeline...", expanded=True) as status:
                enriched = engineer_features(raw_df, include_protocol_dummies=True)
                sample_df = enriched.head(sample_limit).copy()
                features = get_feature_columns(sample_df)
                X = sample_df[features].fillna(0.0)

                t0 = time.time()
                sample_df["Anomaly_IF"] = build_scaled_isolation_forest(contamination=contamination).fit_predict(X)
                t_if = time.time() - t0

                t0 = time.time()
                sample_df["Anomaly_LOF"] = build_scaled_lof(contamination=contamination).fit_predict(X)
                t_lof = time.time() - t0

                bench_rows = [
                    {"Model": "Isolation Forest (Scaled)", "Outliers": int(np.sum(sample_df["Anomaly_IF"] == -1)), "Time_s": round(t_if, 3)},
                    {"Model": "Local Outlier Factor (Scaled)", "Outliers": int(np.sum(sample_df["Anomaly_LOF"] == -1)), "Time_s": round(t_lof, 3)},
                ]

                if include_ocsvm:
                    t0 = time.time()
                    sample_df["Anomaly_OCSVM"] = build_scaled_ocsvm(nu=contamination).fit_predict(X)
                    t_svm = time.time() - t0
                    bench_rows.append({"Model": "One-Class SVM (Scaled)", "Outliers": int(np.sum(sample_df["Anomaly_OCSVM"] == -1)), "Time_s": round(t_svm, 3)})

                st.session_state["v3_df"] = sample_df
                st.session_state["v3_bench"] = pd.DataFrame(bench_rows)
                status.update(label="Pipeline execution complete!", state="complete", expanded=False)

        if "v3_df" in st.session_state:
            sample_df = st.session_state["v3_df"]
            bench_df = st.session_state["v3_bench"]

            st.subheader("📈 Packet Length & Volumetric PPS Timeline")
            tab_t1, tab_t2 = st.tabs(["Packet Sizes", "Packets per Second (DDoS Check)"])
            with tab_t1:
                st.line_chart(data=sample_df.head(1200), x="Time", y="Length", color="#0ea5e9", width="stretch")
            with tab_t2:
                pps = sample_df["Time"].astype(int).value_counts().sort_index()
                st.bar_chart(pps, color="#38bdf8", width="stretch")

            st.subheader("🤖 Multi-Algorithm Benchmarking (RobustScaler Pipelines)")
            cb1, cb2 = st.columns(2)
            with cb1:
                st.bar_chart(data=bench_df, x="Model", y="Outliers", color="#a855f7", width="stretch")
            with cb2:
                st.bar_chart(data=bench_df, x="Model", y="Time_s", color="#10b981", width="stretch")

            # Consensus calculation
            vote_cols = [c for c in ("Anomaly_IF", "Anomaly_LOF", "Anomaly_OCSVM") if c in sample_df.columns]
            sample_df["Consensus_Votes"] = sum((sample_df[c] == -1).astype(int) for c in vote_cols)
            high_conf = sample_df[sample_df["Consensus_Votes"] >= 2].sort_values("Consensus_Votes", ascending=False)

            st.subheader("🚨 High-Confidence Consensus & Dynamic XAI Attribution")
            if not high_conf.empty:
                top_threat = high_conf.iloc[0]
                z_report, z_summary = compute_zscore_attribution(sample_df, top_threat)

                st.markdown(
                    f"""
                    <div class='threat-banner'>
                        <h2 style='color:#fef3c7 !important; margin-top:0;'>⚠️ CONSENSUS ANOMALIES ISOLATED ({len(high_conf)} packet(s) with ≥2 model votes)</h2>
                        <p><b>Top Threat:</b> Packet #{top_threat['No.']} | <b>Votes:</b> {int(top_threat['Consensus_Votes'])}/{len(vote_cols)} |
                        <b>Flow:</b> {top_threat['Source']} → {top_threat['Destination']} | <b>Protocol:</b> {top_threat['Protocol']} ({int(top_threat['Length'])}B)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                cg1, cg2 = st.columns([2, 1])
                with cg1:
                    st.markdown("**Dynamic Z-Score Attribution (Top Anomaly)**")
                    st.markdown(z_summary)
                    st.dataframe(z_report[["Feature", "Baseline_Mean", "Packet_Value", "Z_Score"]].head(8), width="stretch")
                with cg2:
                    st.markdown("**MQTT Mitigation Dispatch**")
                    if live_mqtt:
                        ok, msg = send_mqtt_command("city/alerts/tower", f"EMERGENCY_ON_{top_threat['Source']}", broker_ip=broker_ip)
                        (st.success if ok else st.warning)(msg)
                    else:
                        st.info("Simulation Mode: enable **Auto-send MQTT Alerts** in sidebar or send manually below.")
                        if st.button("🚨 Send Emergency Alert to Tower"):
                            ok, msg = send_mqtt_command("city/alerts/tower", f"EMERGENCY_ON_{top_threat['Source']}", broker_ip=broker_ip)
                            (st.success if ok else st.error)(msg)

                with st.expander("📋 Inspect All Multi-Vote Consensus Packets"):
                    st.dataframe(
                        high_conf[["No.", "Time", "Source", "Destination", "Protocol", "Length", "Consensus_Votes"]].head(50),
                        width="stretch",
                    )
            else:
                st.success("✅ No ≥2-model consensus anomalies detected in this capture window.")
    else:
        st.info("📥 Upload a Wireshark CSV or choose a local CSV dataset above to begin.")

else:
    st.subheader("🌐 Lab Device Registry & Quick MQTT Actuation")
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        device_name = st.selectbox("Select Lab Hardware Node:", list(DEVICES.keys()))
        ip = DEVICES[device_name]
        st.code(f"{device_name} -> {ip}")
        if st.button("🔍 Ping Node"):
            if ping_host(ip):
                st.success(f"✅ {device_name} ({ip}) is online.")
            else:
                st.error(f"❌ {device_name} ({ip}) is unreachable.")

    with col_n2:
        topic = st.selectbox(
            "MQTT Control Topic",
            [
                "city/alerts/tower",
                "city/traffic/control",
                "city/lighting/control",
                "city/watertreatment/master",
                "city/powerstation/turbine",
            ],
        )
        payload = st.text_input("MQTT Payload", value="EMERGENCY_ON")
        if st.button("⚡ Publish MQTT Frame"):
            ok, msg = send_mqtt_command(topic, payload, broker_ip=broker_ip)
            (st.success if ok else st.error)(msg)
