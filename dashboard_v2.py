"""Smart City Autonomous AI Defense Hub (v2 — Fixed & Enhanced)."""

import numpy as np
import pandas as pd
import psutil
import streamlit as st

from smart_city_ml import (
    JETSON_SERVER_IP,
    build_scaled_isolation_forest,
    build_scaled_lof,
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
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 12px; color: white; text-align: center; }
    .threat-box { background-color: #7f1d1d; border-left: 6px solid #ef4444; padding: 20px; border-radius: 8px; color: #fee2e2; }
    .stButton>button { border-radius: 8px; height: 3.5em; font-weight: bold; background-color: #2563eb !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🏙️ Security Control Center")
st.sidebar.write("Turning network telemetry into visual, explainable security decisions.")
st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ Hardware Brain Monitor")
cpu_load = psutil.cpu_percent()
mem_load = psutil.virtual_memory().percent
st.sidebar.write(f"**CPU Core Load:** {cpu_load}%")
st.sidebar.progress(min(cpu_load / 100.0, 1.0))
st.sidebar.write(f"**Memory RAM Load:** {mem_load}%")
st.sidebar.progress(min(mem_load / 100.0, 1.0))

live_mqtt_enabled = st.sidebar.checkbox("Enable Live MQTT Actuation", value=False)

st.title("🛡️ Smart City Autonomous AI Defense Center (v2)")
st.write("Interactive executive dashboard for behavioral feature extraction and multi-model consensus.")
st.markdown("---")

st.header("📥 Stage 1: Data Ingestion Gateway")
uploaded_file = st.file_uploader("Drop your smart city network file here:", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_cached(file_obj) -> pd.DataFrame:
        return load_wireshark_csv(file_obj)

    df = load_cached(uploaded_file)
    st.success("🎉 Data successfully loaded into the AI Core memory matrix!")

    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        st.markdown(
            f"<div class='metric-card'><h3>📊 Ingested Records</h3><h2>{len(df):,}</h2><p>Packet Footprints</p></div>",
            unsafe_allow_html=True,
        )
    with c_in2:
        st.markdown(
            f"<div class='metric-card'><h3>🌐 Active Channels</h3><h2>{df['Protocol'].nunique()}</h2><p>Unique Protocols</p></div>",
            unsafe_allow_html=True,
        )
    with c_in3:
        st.markdown(
            f"<div class='metric-card'><h3>🏢 Device Grid</h3><h2>{df['Source'].nunique()}</h2><p>Communicating Nodes</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.header("🛠️ Stage 2 & 3: Behavioral Feature Extraction & Multi-Model Consensus")

    if st.button("🧠 PROCESS DATA & EVALUATE MODELS", type="primary"):
        with st.status("Computing scaled behavioral representations...", expanded=True) as status:
            enriched = engineer_features(df, include_protocol_dummies=True)
            sample_df = enriched.head(40000).copy()
            features = get_feature_columns(sample_df)
            X = sample_df[features].fillna(0.0)

            iforest = build_scaled_isolation_forest(contamination=0.01, random_state=42)
            sample_df["Anomaly_IF"] = iforest.fit_predict(X)

            lof = build_scaled_lof(contamination=0.01, n_neighbors=20)
            sample_df["Anomaly_LOF"] = lof.fit_predict(X)

            st.session_state["v2_sample_df"] = sample_df
            status.update(label="Feature extraction & model scoring complete!", state="complete", expanded=False)

    if "v2_sample_df" in st.session_state:
        sample_df = st.session_state["v2_sample_df"]

        st.subheader("📈 Live Traffic Ingestion Pattern (Chronological Time-Series)")
        st.line_chart(data=sample_df.head(1000), x="Time", y="Length", width="stretch")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.subheader("🌲 Brain A: Scaled Isolation Forest")
            st.metric("Suspect Points Flagged", f"{int(np.sum(sample_df['Anomaly_IF'] == -1))} packets")
        with col_b2:
            st.subheader("🌐 Brain B: Scaled Local Outlier Factor")
            st.metric("Suspect Points Flagged", f"{int(np.sum(sample_df['Anomaly_LOF'] == -1))} packets")

        st.markdown("---")
        st.header("🚨 Stage 4: Joint Consensus Agreement & Defense Response")

        consensus_df = sample_df[(sample_df["Anomaly_IF"] == -1) & (sample_df["Anomaly_LOF"] == -1)]

        if not consensus_df.empty:
            threat = consensus_df.iloc[0]
            z_report, z_summary = compute_zscore_attribution(sample_df, threat)

            st.markdown(
                f"""
                <div class='threat-box'>
                    <h2>🔥 MULTI-MODEL CONSENSUS ANOMALY DETECTED ({len(consensus_df)} packet(s))</h2>
                    <p><b>Threat Fingerprint:</b> Both scaled anomaly engines independently flagged this packet.</p>
                    <ul>
                        <li><b>Source Node:</b> {threat['Source']}</li>
                        <li><b>Destination Node:</b> {threat['Destination']}</li>
                        <li><b>Protocol:</b> {threat['Protocol']} ({int(threat['Length'])} bytes)</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_g1, col_g2 = st.columns([2, 1])
            with col_g1:
                st.subheader("📊 Outlier Density by Protocol Layer")
                all_anomalies = sample_df[(sample_df["Anomaly_IF"] == -1) | (sample_df["Anomaly_LOF"] == -1)]
                st.bar_chart(all_anomalies["Protocol"].value_counts(), color="#ef4444")

            with col_g2:
                st.subheader("🛡️ Defense Mitigation Status")
                if live_mqtt_enabled:
                    ok, msg = send_mqtt_command("city/alerts/tower", f"EMERGENCY_ON_{threat['Source']}", broker_ip=JETSON_SERVER_IP)
                    if ok:
                        st.success(f"🔒 {msg}")
                    else:
                        st.warning(f"⚠️ {msg}")
                else:
                    st.info("ℹ️ Simulation Mode active. Enable **Live MQTT Actuation** in the sidebar or click below to dispatch real alerts.")
                    if st.button("🚨 Dispatch MQTT Alert Tower Command"):
                        ok, msg = send_mqtt_command("city/alerts/tower", f"EMERGENCY_ON_{threat['Source']}", broker_ip=JETSON_SERVER_IP)
                        (st.success if ok else st.error)(msg)

            with st.expander("👁️ Dynamic Explainable AI (XAI) Attribution"):
                st.markdown(z_summary)
                st.dataframe(z_report[["Feature", "Baseline_Mean", "Packet_Value", "Z_Score"]], width="stretch")
        else:
            st.success("✅ System Status: Secure. No multi-brain consensus anomalies found inside this segment.")
else:
    st.info("💡 Drag and drop a network traffic CSV file above to start analysis.")
