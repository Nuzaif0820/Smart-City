"""Smart City Autonomous AI Defense Hub (v3 Flagship — Cyber SOC UI)."""

import os
import time
from pathlib import Path
from typing import Dict, List

import altair as alt
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

st.set_page_config(
    page_title="Smart City AI Defense Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CYBER-SOC CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 12% 8%, rgba(6, 182, 212, 0.09) 0%, transparent 35%),
                    radial-gradient(circle at 88% 12%, rgba(99, 102, 241, 0.09) 0%, transparent 38%),
                    #060b16;
    }

    /* Hero Banner */
    .hero-shell {
        background: linear-gradient(135deg, rgba(13, 21, 38, 0.95) 0%, rgba(15, 28, 52, 0.92) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 55%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }
    .status-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 7px 14px;
        border-radius: 999px;
        letter-spacing: 0.04em;
        display: inline-block;
    }
    .pill-sim {
        background: rgba(6, 182, 212, 0.14);
        border: 1px solid rgba(6, 182, 212, 0.45);
        color: #67e8f9;
    }
    .pill-live {
        background: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
    }

    /* Glass KPI Cards */
    .kpi-card {
        background: rgba(13, 21, 38, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.45);
    }
    .kpi-accent {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.95rem;
        font-weight: 800;
        line-height: 1.15;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-foot {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Threat Banner */
    .threat-banner {
        background: linear-gradient(120deg, rgba(127, 29, 29, 0.55) 0%, rgba(153, 27, 27, 0.35) 55%, rgba(69, 26, 3, 0.55) 100%);
        border: 1px solid rgba(248, 113, 113, 0.45);
        border-left: 6px solid #ef4444;
        border-radius: 14px;
        padding: 20px 24px;
        color: #fee2e2;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.16);
        margin: 12px 0 20px 0;
    }
    .threat-banner h3 {
        margin: 0 0 8px 0;
        color: #fecaca;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .threat-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #fde68a;
    }

    /* Safe Banner */
    .safe-banner {
        background: linear-gradient(120deg, rgba(6, 78, 59, 0.42) 0%, rgba(13, 21, 38, 0.85) 100%);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 18px 22px;
        color: #d1fae5;
        margin: 12px 0 20px 0;
    }

    /* Sector Card */
    .sector-card {
        background: rgba(13, 21, 38, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .node-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.14);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.35);
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: #ffffff;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(6, 182, 212, 0.35);
        border-color: #67e8f9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_html(label: str, value: str, footer: str, accent: str) -> str:
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-accent' style='background:{accent};'></div>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-foot'>{footer}</div>"
        f"</div>"
    )


def style_altair(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_axis(
            labelColor="#94a3b8",
            titleColor="#cbd5e1",
            gridColor="rgba(148,163,184,0.12)",
            domainColor="rgba(148,163,184,0.25)",
        )
        .configure_legend(labelColor="#cbd5e1", titleColor="#f8fafc")
        .configure_title(color="#f8fafc", fontSize=14, anchor="start")
    )


# --- SIDEBAR ---
st.sidebar.markdown("## 🛡️ Sentinel SOC")
st.sidebar.caption("Smart City CPS Anomaly & Actuation Console")
st.sidebar.markdown("---")

nav = st.sidebar.radio(
    "Navigation",
    ["🛡️ Threat Detection & XAI", "🏭 Lab Topology & MQTT Controls"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔌 Broker & Actuation")
broker_ip = st.sidebar.text_input("Jetson MQTT IP", value=JETSON_SERVER_IP)
live_mqtt = st.sidebar.toggle("Auto-Dispatch MQTT Alerts", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 Host Telemetry")
cpu = psutil.cpu_percent()
ram = psutil.virtual_memory().percent
disk = psutil.disk_usage(os.path.abspath(os.sep)).percent
st.sidebar.caption(f"CPU `{cpu:.0f}%`  ·  RAM `{ram:.0f}%`  ·  Disk `{disk:.0f}%`")
st.sidebar.progress(min(cpu / 100.0, 1.0))

# --- HERO HEADER ---
mode_badge = (
    "<span class='status-pill pill-live'>⚡ LIVE MQTT ARMED</span>"
    if live_mqtt
    else "<span class='status-pill pill-sim'>🧪 SIMULATION MODE</span>"
)
st.markdown(
    f"""
    <div class='hero-shell'>
        <div>
            <h1 class='hero-title'>🏙️ Smart City Autonomous AI Defense Center</h1>
            <p class='hero-sub'>Scaled multi-algorithm packet intelligence, dynamic Z-score XAI attribution, and closed-loop CPS protection.</p>
        </div>
        <div>{mode_badge}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# VIEW 1: THREAT DETECTION & XAI
# =====================================================================
if nav == "🛡️ Threat Detection & XAI":
    with st.container():
        col_src1, col_src2 = st.columns([1.4, 1.0])
        with col_src1:
            uploaded_file = st.file_uploader("📥 Upload Wireshark Capture CSV", type=["csv"])
        with col_src2:
            local_csvs = sorted(
                p.name for p in BASE_DIR.glob("smart_city_*.csv") if p.stat().st_size < 50_000_000
            )
            default_idx = (
                local_csvs.index("smart_city_traffic_light.csv") + 1
                if "smart_city_traffic_light.csv" in local_csvs
                else (1 if local_csvs else 0)
            )
            selected_local = st.selectbox(
                "🗂️ Or Quick-Load Workspace Dataset (<50 MB)",
                ["(None)"] + local_csvs,
                index=default_idx,
            )

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
            raw_df["Source"].str.startswith("172.16.16.")
            | raw_df["Destination"].str.startswith("172.16.16.")
        ).mean() * 100.0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(kpi_html("Ingested Packets", f"{len(raw_df):,}", dataset_label or "", "linear-gradient(90deg,#06b6d4,#38bdf8)"), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_html("Unique Protocols", f"{raw_df['Protocol'].nunique()}", f"Top: {raw_df['Protocol'].mode().iloc[0]}", "linear-gradient(90deg,#8b5cf6,#a855f7)"), unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_html("Active Endpoints", f"{raw_df['Source'].nunique()}", "Distinct Source Nodes", "linear-gradient(90deg,#10b981,#34d399)"), unsafe_allow_html=True)
        with k4:
            st.markdown(kpi_html("Lab Subnet Share", f"{lab_ratio:.1f}%", "172.16.16.0/24 OT/IoT", "linear-gradient(90deg,#f59e0b,#fbbf24)"), unsafe_allow_html=True)

        with st.expander("⚙️ Model & Window Configuration", expanded=True):
            c_opt1, c_opt2, c_opt3, c_opt4 = st.columns([1.2, 1.2, 1.0, 1.2])
            sample_limit = c_opt1.slider(
                "Evaluation Window", 500, 40000, min(15000, len(raw_df)), step=500
            )
            contamination = c_opt2.slider("Contamination Rate", 0.005, 0.05, 0.01, step=0.005)
            include_ocsvm = c_opt3.checkbox("Include One-Class SVM", value=True)
            with c_opt4:
                st.write("")
                run_clicked = st.button("🚀 RUN AI DEFENSE SCAN", width="stretch")

        if run_clicked:
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
                    {"Model": "Isolation Forest", "Outliers": int(np.sum(sample_df["Anomaly_IF"] == -1)), "Time_s": round(t_if, 3)},
                    {"Model": "Local Outlier Factor", "Outliers": int(np.sum(sample_df["Anomaly_LOF"] == -1)), "Time_s": round(t_lof, 3)},
                ]

                if include_ocsvm:
                    t0 = time.time()
                    sample_df["Anomaly_OCSVM"] = build_scaled_ocsvm(nu=contamination).fit_predict(X)
                    t_svm = time.time() - t0
                    bench_rows.append({"Model": "One-Class SVM", "Outliers": int(np.sum(sample_df["Anomaly_OCSVM"] == -1)), "Time_s": round(t_svm, 3)})

                vote_cols = [c for c in ("Anomaly_IF", "Anomaly_LOF", "Anomaly_OCSVM") if c in sample_df.columns]
                sample_df["Consensus_Votes"] = sum((sample_df[c] == -1).astype(int) for c in vote_cols)

                st.session_state["v3_df"] = sample_df
                st.session_state["v3_bench"] = pd.DataFrame(bench_rows)
                st.session_state["v3_dataset"] = dataset_label
                status.update(label="Multi-model defense scan complete!", state="complete", expanded=False)

        if "v3_df" in st.session_state and st.session_state.get("v3_dataset") == dataset_label:
            sample_df = st.session_state["v3_df"]
            bench_df = st.session_state["v3_bench"]
            vote_cols = [c for c in ("Anomaly_IF", "Anomaly_LOF", "Anomaly_OCSVM") if c in sample_df.columns]
            high_conf = sample_df[sample_df["Consensus_Votes"] >= 2].sort_values(
                ["Consensus_Votes", "Length"], ascending=False
            )

            # Consensus Banner
            if not high_conf.empty:
                top_threat = high_conf.iloc[0]
                st.markdown(
                    f"""
                    <div class='threat-banner'>
                        <h3>🚨 MULTI-MODEL CONSENSUS ISOLATED ({len(high_conf)} packet(s) with ≥2 votes)</h3>
                        <div class='threat-meta'>
                            Top Hit: Packet #{top_threat['No.']} &nbsp;|&nbsp;
                            Votes: {int(top_threat['Consensus_Votes'])}/{len(vote_cols)} &nbsp;|&nbsp;
                            {top_threat['Source']} → {top_threat['Destination']} &nbsp;|&nbsp;
                            {top_threat['Protocol']} ({int(top_threat['Length'])} B)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='safe-banner'><b>✅ NOMINAL TRAFFIC STATUS:</b> No ≥2-model consensus anomalies flagged in this window.</div>",
                    unsafe_allow_html=True,
                )

            # Tabs for visual clarity
            tab_overview, tab_models, tab_xai, tab_packets = st.tabs([
                "📡 Telemetry & DDoS",
                "🤖 Model Benchmarks",
                "🔮 Explainable AI (XAI)",
                "📋 Anomaly Explorer",
            ])

            with tab_overview:
                plot_df = sample_df.head(1500).copy()
                base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Capture Time (s)"))
                area_line = base.mark_area(
                    line={"color": "#38bdf8", "strokeWidth": 1.6},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="rgba(56,189,248,0.35)", offset=0),
                            alt.GradientStop(color="rgba(56,189,248,0.02)", offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                ).encode(
                    y=alt.Y("Length:Q", title="Packet Size (Bytes)"),
                    tooltip=["No.", "Time", "Source", "Destination", "Protocol", "Length", "Consensus_Votes"],
                )

                anomaly_pts = plot_df[plot_df["Consensus_Votes"] >= 2]
                overlay = alt.Chart(anomaly_pts).mark_circle(size=95, color="#ef4444", opacity=0.95, stroke="#fecaca", strokeWidth=1.5).encode(
                    x="Time:Q",
                    y="Length:Q",
                    tooltip=["No.", "Time", "Source", "Destination", "Protocol", "Length", "Consensus_Votes"],
                )
                st.altair_chart(
                    style_altair((area_line + overlay).properties(height=300, title="Packet Size Timeline with Consensus Anomaly Overlays")).interactive(),
                    width="stretch",
                )

                c_ov1, c_ov2 = st.columns(2)
                with c_ov1:
                    pps = sample_df["Time"].astype(int).value_counts().sort_index().reset_index()
                    pps.columns = ["Second", "Packets"]
                    pps_chart = (
                        alt.Chart(pps)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#06b6d4")
                        .encode(
                            x=alt.X("Second:O", title="Capture Second"),
                            y=alt.Y("Packets:Q", title="Packets / Sec"),
                            tooltip=["Second", "Packets"],
                        )
                        .properties(height=250, title="Volumetric Rate (Packets per Second)")
                    )
                    st.altair_chart(style_altair(pps_chart), width="stretch")

                with c_ov2:
                    proto_df = sample_df["Protocol"].value_counts().head(7).reset_index()
                    proto_df.columns = ["Protocol", "Count"]
                    donut = (
                        alt.Chart(proto_df)
                        .mark_arc(innerRadius=55, outerRadius=95)
                        .encode(
                            theta=alt.Theta("Count:Q"),
                            color=alt.Color("Protocol:N", scale=alt.Scale(scheme="tableau10")),
                            tooltip=["Protocol", "Count"],
                        )
                        .properties(height=250, title="Protocol Layer Mix")
                    )
                    st.altair_chart(style_altair(donut), width="stretch")

            with tab_models:
                cm1, cm2 = st.columns(2)
                with cm1:
                    c_out = (
                        alt.Chart(bench_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("Model:N", title=""),
                            y=alt.Y("Outliers:Q", title="Flagged Packets"),
                            color=alt.Color("Model:N", scale=alt.Scale(range=["#38bdf8", "#a855f7", "#f59e0b"]), legend=None),
                            tooltip=["Model", "Outliers", "Time_s"],
                        )
                        .properties(height=270, title="Outlier Count by Scaled Model")
                    )
                    st.altair_chart(style_altair(c_out), width="stretch")

                with cm2:
                    c_spd = (
                        alt.Chart(bench_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("Model:N", title=""),
                            y=alt.Y("Time_s:Q", title="Execution Time (s)"),
                            color=alt.Color("Model:N", scale=alt.Scale(range=["#10b981", "#06b6d4", "#6366f1"]), legend=None),
                            tooltip=["Model", "Time_s", "Outliers"],
                        )
                        .properties(height=270, title="Compute Latency Comparison")
                    )
                    st.altair_chart(style_altair(c_spd), width="stretch")

                st.dataframe(bench_df, width="stretch", hide_index=True)

            with tab_xai:
                target_pool = high_conf if not high_conf.empty else sample_df[sample_df["Consensus_Votes"] >= 1]
                if not target_pool.empty:
                    target_labels = [
                        f"Packet #{int(r['No.'])} · {r['Protocol']} · {r['Source']} → {r['Destination']} ({int(r['Consensus_Votes'])} votes)"
                        for _, r in target_pool.head(25).iterrows()
                    ]
                    picked = st.selectbox("Select Flagged Packet to Diagnose:", range(len(target_labels)), format_func=lambda i: target_labels[i])
                    chosen_row = target_pool.iloc[picked]
                    z_report, z_summary = compute_zscore_attribution(sample_df, chosen_row)

                    cx1, cx2 = st.columns([1.5, 1.0])
                    with cx1:
                        z_chart_df = z_report.copy()
                        z_chart_df["Direction"] = np.where(z_chart_df["Z_Score"] >= 0, "Above Baseline", "Below Baseline")
                        z_bar = (
                            alt.Chart(z_chart_df)
                            .mark_bar(cornerRadiusEnd=4)
                            .encode(
                                y=alt.Y("Feature:N", sort="-x", title=""),
                                x=alt.X("Z_Score:Q", title="Standard Deviations (σ)"),
                                color=alt.Color("Direction:N", scale=alt.Scale(domain=["Above Baseline", "Below Baseline"], range=["#ef4444", "#38bdf8"])),
                                tooltip=["Feature", "Baseline_Mean", "Packet_Value", "Z_Score"],
                            )
                            .properties(height=300, title="Feature Attribution (Z-Score Deviation)")
                        )
                        st.altair_chart(style_altair(z_bar), width="stretch")

                    with cx2:
                        st.markdown("#### 🧠 AI Diagnosis")
                        st.info(z_summary)
                        st.markdown("#### 🛡️ Mitigation Dispatch")
                        if live_mqtt:
                            ok, msg = send_mqtt_command("city/alerts/tower", f"EMERGENCY_ON_{chosen_row['Source']}", broker_ip=broker_ip)
                            (st.success if ok else st.warning)(msg)
                        else:
                            if st.button("🚨 Trigger Alert Tower (MQTT)", width="stretch"):
                                ok, msg = send_mqtt_command("city/alerts/tower", f"EMERGENCY_ON_{chosen_row['Source']}", broker_ip=broker_ip)
                                (st.success if ok else st.error)(msg)
                        st.dataframe(z_report[["Feature", "Packet_Value", "Z_Score"]].head(6), width="stretch", hide_index=True)
                else:
                    st.info("No flagged packets available in this window for XAI breakdown.")

            with tab_packets:
                view_cols = [
                    c for c in [
                        "No.", "Time", "Source", "Destination", "Protocol", "Length",
                        "Consensus_Votes", "Rolling_Source_Burst", "IAT_Variance", "Is_Lab_Subnet",
                    ]
                    if c in sample_df.columns
                ]
                flagged_only = st.checkbox("Show only packets with ≥1 model vote", value=True)
                table_df = sample_df[sample_df["Consensus_Votes"] >= 1] if flagged_only else sample_df
                st.dataframe(
                    table_df.sort_values("Consensus_Votes", ascending=False)[view_cols].head(200),
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.info("📥 Upload a Wireshark CSV capture or select a local workspace dataset above to launch the scan.")

# =====================================================================
# VIEW 2: LAB TOPOLOGY & MQTT CONTROLS
# =====================================================================
else:
    st.subheader("🌐 Smart City Physical Hardware Registry & Diagnostics")
    col_t1, col_t2 = st.columns([1.15, 1.0])

    with col_t1:
        st.markdown("<div class='sector-card'>", unsafe_allow_html=True)
        selected_dev = st.selectbox("Choose Lab Endpoint", list(DEVICES.keys()))
        dev_ip = DEVICES[selected_dev]
        hw_tag = (
            "Siemens PLC" if "PLC" in selected_dev else
            "Raspberry Pi" if "(Pi)" in selected_dev else
            "ESP8266" if "ESP8266" in selected_dev else
            "Arduino" if "Arduino" in selected_dev else
            "Core Infrastructure"
        )
        st.markdown(f"<span class='node-badge'>{hw_tag} · {dev_ip}</span>", unsafe_allow_html=True)

        c_ping, c_scan = st.columns(2)
        with c_ping:
            if st.button("🔍 Ping Selected Node", width="stretch"):
                if ping_host(dev_ip):
                    st.success(f"✅ {selected_dev} ({dev_ip}) responded.")
                else:
                    st.error(f"❌ {selected_dev} ({dev_ip}) unreachable.")
        with c_scan:
            if st.button("⚡ Ping Jetson Broker", width="stretch"):
                if ping_host(broker_ip):
                    st.success(f"✅ Jetson MQTT broker ({broker_ip}) online.")
                else:
                    st.error(f"❌ Jetson MQTT broker ({broker_ip}) unreachable.")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📋 Full Lab Device Directory (38 Nodes)"):
            dir_df = pd.DataFrame([{"Node": k, "IP Address": v} for k, v in DEVICES.items()])
            st.dataframe(dir_df, width="stretch", hide_index=True)

    with col_t2:
        st.markdown("<div class='sector-card'>", unsafe_allow_html=True)
        st.markdown("#### 🎛️ Sector Quick-Actuation Panel")
        presets: Dict[str, tuple[str, str]] = {
            "🚨 Activate Alert Towers": ("city/alerts/tower", "EMERGENCY_ON"),
            "🚦 Force All Traffic Signals RED": ("city/traffic/control", "RED"),
            "💡 Turn ON Street Lighting": ("city/lighting/control", "ON"),
            "🚰 Stop Water Treatment Plant": ("city/watertreatment/master", "STOP"),
            "⚡ Trip Power Station Reset": ("city/powerstation/reset", "RESET"),
            "🚇 Switch Metro Track": ("city/metro/track", "SWITCH"),
        }
        p_cols = st.columns(2)
        for idx, (btn_label, (topic, payload)) in enumerate(presets.items()):
            with p_cols[idx % 2]:
                if st.button(btn_label, width="stretch"):
                    ok, msg = send_mqtt_command(topic, payload, broker_ip=broker_ip)
                    (st.success if ok else st.error)(msg)

        st.markdown("---")
        custom_topic = st.text_input("Custom MQTT Topic", value="city/alerts/tower")
        custom_payload = st.text_input("Custom Payload", value="EMERGENCY_ON")
        if st.button("📡 Send Custom MQTT Frame", width="stretch"):
            ok, msg = send_mqtt_command(custom_topic, custom_payload, broker_ip=broker_ip)
            (st.success if ok else st.error)(msg)
        st.markdown("</div>", unsafe_allow_html=True)
