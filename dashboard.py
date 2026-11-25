"""Smart City Operational Dashboard (v1): Node Registry, ML Core, and MQTT Controls."""

import os

import psutil
import streamlit as st

from health_check import ping_host
from smart_city_ml import (
    BASE_DIR,
    DEVICES,
    JETSON_SERVER_IP,
    build_scaled_isolation_forest,
    compute_zscore_attribution,
    engineer_features,
    get_feature_columns,
    load_wireshark_csv,
    send_mqtt_command,
)

st.set_page_config(page_title="Smart City Security Platform", page_icon="🏙️", layout="wide")


def dispatch_mqtt(topic: str, payload: str) -> None:
    ok, msg = send_mqtt_command(topic, payload, broker_ip=JETSON_SERVER_IP)
    if ok:
        st.success(f"⚡ {msg}")
    else:
        st.error(f"❌ {msg}")


st.sidebar.title("🎮 Operations Hub")
page = st.sidebar.radio(
    "Navigate System:",
    [
        "🖥️ Node Registry & Ping Scanner",
        "🤖 Machine Learning Core",
        "🚨 Master Smart City Controls",
    ],
)

if page == "🖥️ Node Registry & Ping Scanner":
    st.title("🖥️ Real-Time Cyber-Physical Infrastructure Registry")
    st.write("Verifies host resource status and connectivity to the table network nodes.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📊 Host Resource Allocation")
        st.metric("CPU Utilization", f"{psutil.cpu_percent(interval=0.1)}%")
        st.metric("System RAM Load", f"{psutil.virtual_memory().percent}%")
        st.metric("Local Storage Used", f"{psutil.disk_usage(os.path.abspath(os.sep)).percent}%")

    with col2:
        st.subheader("🌐 Network Device Mapping Grid")
        target_name = st.selectbox("Choose Hardware Endpoint Target:", list(DEVICES.keys()))
        target_ip = DEVICES[target_name]
        st.info(f"Target Static IP Profile: `{target_ip}`")

        if st.button("🔍 Execute Network Diagnostic Ping"):
            with st.spinner(f"Pinging {target_name}..."):
                if ping_host(target_ip):
                    st.success(f"✅ Connection Online! Node {target_name} at `{target_ip}` responded.")
                else:
                    st.error(f"❌ Endpoint Offline! Device at `{target_ip}` did not respond.")

elif page == "🤖 Machine Learning Core":
    st.title("🤖 ML Autonomous Anomaly Core")
    st.write("Upload any Wireshark capture CSV to run scaled Isolation Forest anomaly detection.")
    st.markdown("---")

    uploaded_file = st.file_uploader("📥 Drag and drop your network dataset here:", type=["csv"])
    contamination = st.slider("System Contamination Threshold (Outlier Ratio)", 0.005, 0.05, 0.01, step=0.005)

    if uploaded_file is not None:
        if st.button("🚀 INGEST NETWORK TRAFFIC & TRAIN SYSTEM", type="primary"):
            with st.spinner("Processing network stream records..."):
                try:
                    df = load_wireshark_csv(uploaded_file)
                    df = engineer_features(df, include_protocol_dummies=True)
                    features = get_feature_columns(df)
                    X = df[features].fillna(0.0)

                    clf = build_scaled_isolation_forest(contamination=contamination, random_state=42)
                    df["Anomaly_Score"] = clf.fit_predict(X)
                    df.to_csv(BASE_DIR / "smart_city_anomalies_predicted.csv", index=False)

                    anomalies = df[df["Anomaly_Score"] == -1]
                    st.success("🤖 Scaled Isolation Forest evaluation finalized!")
                    c1, c2 = st.columns(2)
                    c1.metric("Ingested Packets Analyzed", f"{len(df):,}")
                    c2.metric("Extracted Outliers", f"{len(anomalies):,}")

                    if not anomalies.empty:
                        st.subheader("📈 Captured Outlier Incidents by Protocol Layer")
                        st.bar_chart(anomalies["Protocol"].value_counts())

                        _, summary = compute_zscore_attribution(df, anomalies.iloc[0])
                        st.info(summary)

                        st.subheader("📋 Top Identified Anomaly Vectors")
                        st.dataframe(
                            anomalies[["No.", "Time", "Source", "Destination", "Protocol", "Length"]].head(15),
                            width="stretch",
                        )
                except Exception as exc:
                    st.error(f"❌ Processing Error: {exc}")
    else:
        st.info("💡 Upload a network traffic CSV file above to initialize the AI engine pipeline.")

elif page == "🚨 Master Smart City Controls":
    st.title("🚨 Master Operational & Cyber-Physical Intervention Console")
    st.write("Directly inject MQTT control frames into physical sectors across the lab bench network.")
    st.markdown("---")

    st.header("⚙️ Sector A: Heavy Utilities & Industrial Plant Control Matrix")
    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.subheader("🚰 Water Treatment Plant")
        st.caption("Siemens PLC Target Node: `172.16.16.122`")
        if st.button("🔄 Activate Combined Plant Filtration"):
            dispatch_mqtt("city/watertreatment/master", "START")
        if st.button("🛑 Force Complete Safety Cutoff"):
            dispatch_mqtt("city/watertreatment/master", "STOP")

    with col_a2:
        st.subheader("⚡ Power Grid Substation & Turbine")
        st.caption("Siemens PLC Target Node: `172.16.16.121`")
        if st.button("🚀 Fire Up Generation Turbine"):
            dispatch_mqtt("city/powerstation/turbine", "ON")
        if st.button("🔌 Close Substation Grid Loop"):
            dispatch_mqtt("city/powerstation/substation", "CLOSE")
        if st.button("🔄 Emergency System Grid Reset"):
            dispatch_mqtt("city/powerstation/reset", "RESET")

    with col_a3:
        st.subheader("🏢 Mission Critical Data Center")
        st.caption("Siemens PLC Target Node: `172.16.16.126`")
        if st.button("❄️ Activate DC Coolant Pumps"):
            dispatch_mqtt("city/datacenter/coolant", "ON")
        if st.button("💨 Override HVAC Ventilation Fans"):
            dispatch_mqtt("city/datacenter/ventilation", "ON")

    st.markdown("---")
    st.header("🚄 Sector B: Public Transit Logistics & Automated Highways")
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        st.subheader("🛄 Airport Airside Automation")
        st.caption("Siemens PLCs: `.128`, `.130`")
        if st.button("📦 Start Check-in Conveyors"):
            dispatch_mqtt("city/airport/checkin/conveyor", "RUN")
        if st.button("🛬 Start Baggage Claim Carousel"):
            dispatch_mqtt("city/airport/claim/carousel", "RUN")
        if st.button("🚪 Override Security Gates"):
            dispatch_mqtt("city/airport/gates/override", "OPEN")

    with col_b2:
        st.subheader("🎫 Intelligent RFID Toll Infrastructure")
        st.caption("Arduinos: `.101`, `.102`")
        if st.button("🔓 Force Open Tollgate 1"):
            dispatch_mqtt("city/tollgate1/barrier", "OPEN")
        if st.button("🔓 Force Open Tollgate 2"):
            dispatch_mqtt("city/tollgate2/barrier", "OPEN")
        if st.button("🔄 Reset Tollplaza Controllers"):
            dispatch_mqtt("city/tollgate/master/reset", "RESET")

    with col_b3:
        st.subheader("🚇 Municipal Metro Rail System")
        st.caption("Siemens PLC: `172.16.16.127`")
        if st.button("🔀 Switch Line Intersect Track"):
            dispatch_mqtt("city/metro/track", "SWITCH")

    st.markdown("---")
    st.header("🏙️ Sector C: Civil Management & Cyber Incident Containment")
    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        st.subheader("🚦 Highway Intersections")
        st.caption("Raspberry Pi: `172.16.16.109`")
        if st.button("💡 Turn ON Street Lights"):
            dispatch_mqtt("city/lighting/control", "ON")
        if st.button("🛑 Force All Signals RED"):
            dispatch_mqtt("city/traffic/control", "RED")

    with col_c2:
        st.subheader("🏡 Micro-Grid Smart Meters")
        st.caption("ESP8266 Nodes: `.118`, `.119`")
        if st.button("💳 Voucher: Villa 1 Smart Meter"):
            dispatch_mqtt("city/smartmeter1/recharge", "1000")
        if st.button("💳 Voucher: Villa 2 Smart Meter"):
            dispatch_mqtt("city/smartmeter2/recharge", "1000")

    with col_c3:
        st.subheader("💥 Security Incident Mitigation")
        st.caption("Alert Towers: `172.16.16.132`")
        if st.button("🚨 ACTIVATE HARDWARE ALERT TOWERS", type="primary"):
            dispatch_mqtt("city/alerts/tower", "EMERGENCY_ON")
