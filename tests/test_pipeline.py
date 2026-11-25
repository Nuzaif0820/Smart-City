"""Regression & parity tests for the Smart City ML & CPS Defense platform."""

import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scapy.all as scapy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_tap_sniffer import classify_packet_protocol
from smart_city_ml import (
    BASE_DIR,
    LIVE_CORE_FEATURES,
    LiveWindowExtractor,
    build_scaled_isolation_forest,
    build_scaled_lof,
    build_scaled_ocsvm,
    compute_zscore_attribution,
    engineer_features,
    get_feature_columns,
)


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        {"No.": 1, "Time": 0.000, "Source": "172.16.16.100", "Destination": "172.16.16.109", "Protocol": "MQTT", "Length": 84, "Info": "Publish"},
        {"No.": 2, "Time": 0.015, "Source": "172.16.16.109", "Destination": "224.0.0.251", "Protocol": "MDNS", "Length": 112, "Info": "Standard query"},
        {"No.": 3, "Time": 0.025, "Source": "172.16.16.109", "Destination": "224.0.0.251", "Protocol": "MDNS", "Length": 118, "Info": "Standard query"},
        {"No.": 4, "Time": 0.040, "Source": "172.16.16.101", "Destination": "Broadcast", "Protocol": "ARP", "Length": 60, "Info": "Who has 172.16.16.100"},
        {"No.": 5, "Time": 0.120, "Source": "172.16.17.43", "Destination": "20.184.175.4", "Protocol": "TLSv1.2", "Length": 1514, "Info": "Application Data"},
    ])


class TestSmartCityPipeline(unittest.TestCase):
    def test_protocol_dummies_are_preserved(self) -> None:
        df = _sample_dataframe()
        enriched = engineer_features(df, include_protocol_dummies=True)
        p_cols = [c for c in enriched.columns if c.startswith("P_")]
        self.assertIn("P_MQTT", p_cols)
        self.assertIn("P_MDNS", p_cols)
        self.assertIn("P_ARP", p_cols)
        self.assertEqual(enriched["P_MDNS"].sum(), 2)
        self.assertEqual(enriched["Rolling_MDNS_Count"].iloc[-1], 2)

    def test_batch_and_live_feature_parity(self) -> None:
        df = _sample_dataframe()
        enriched = engineer_features(df, include_protocol_dummies=True)
        cols = get_feature_columns(enriched, core_features=LIVE_CORE_FEATURES)

        extractor = LiveWindowExtractor(max_buffer_size=50, expected_columns=cols)
        live_rows = []
        for _, row in df.iterrows():
            live_df = extractor.update(
                p_time=row["Time"],
                p_length=row["Length"],
                proto_label=row["Protocol"],
                src_ip=row["Source"],
                dst_ip=row["Destination"],
            )
            live_rows.append(live_df.iloc[0].to_dict())

        live_full = pd.DataFrame(live_rows)[cols]
        batch_full = enriched[cols].reset_index(drop=True)

        for col in cols:
            self.assertTrue(
                np.allclose(live_full[col].values, batch_full[col].values, atol=1e-9),
                f"Mismatch in {col}",
            )

    def test_live_sniffer_detects_mdns_and_mqtt(self) -> None:
        mdns_pkt = scapy.IP(src="172.16.16.109", dst="224.0.0.251") / scapy.UDP(sport=5353, dport=5353)
        mqtt_pkt = scapy.IP(src="172.16.16.100", dst="172.16.16.109") / scapy.TCP(sport=1883, dport=51234)
        arp_pkt = scapy.ARP(psrc="172.16.16.101", pdst="172.16.16.100")

        self.assertEqual(classify_packet_protocol(mdns_pkt), "MDNS")
        self.assertEqual(classify_packet_protocol(mqtt_pkt), "MQTT")
        self.assertEqual(classify_packet_protocol(arp_pkt), "ARP")

    def test_scaled_models_respect_contamination_on_synthetic_stream(self) -> None:
        rng = np.random.default_rng(42)
        n_rows = 400
        df = pd.DataFrame({
            "No.": np.arange(1, n_rows + 1),
            "Time": np.cumsum(rng.uniform(0.001, 0.02, size=n_rows)),
            "Source": ["172.16.16.100" if i % 2 == 0 else "172.16.16.109" for i in range(n_rows)],
            "Destination": ["172.16.16.109" if i % 2 == 0 else "172.16.16.100" for i in range(n_rows)],
            "Protocol": ["TCP"] * n_rows,
            "Length": rng.normal(150, 20, size=n_rows).clip(54, 400),
        })
        enriched = engineer_features(df, include_protocol_dummies=True)
        X = enriched[get_feature_columns(enriched)].fillna(0.0)

        if_preds = build_scaled_isolation_forest(contamination=0.02).fit_predict(X)
        lof_preds = build_scaled_lof(contamination=0.02).fit_predict(X)
        svm_preds = build_scaled_ocsvm(nu=0.02).fit_predict(X)

        self.assertTrue(1 <= int((if_preds == -1).sum()) <= 20)
        self.assertTrue(1 <= int((lof_preds == -1).sum()) <= 20)
        self.assertTrue(1 <= int((svm_preds == -1).sum()) <= 25)

    def test_dynamic_xai_identifies_actual_outlier_driver(self) -> None:
        rows = [
            {
                "No.": i + 1,
                "Time": i * 0.01,
                "Source": "172.16.16.100",
                "Destination": "172.16.16.109",
                "Protocol": "TCP",
                "Length": 80,
                "Info": "Normal",
            }
            for i in range(20)
        ]
        rows[-1]["Length"] = 1514
        enriched = engineer_features(pd.DataFrame(rows), include_protocol_dummies=True)
        target = enriched.iloc[-1]
        report, summary = compute_zscore_attribution(enriched, target)
        self.assertIn(report.iloc[0]["Feature"], {"Length", "Rolling_Mean_Length", "Rolling_Std_Length"})
        self.assertIn("Flagged packet", summary)

    def test_dashboards_have_valid_syntax_and_no_unsafe_with_html_typo(self) -> None:
        for fname in ("dashboard.py", "dashboard_v2.py", "dashboard_v3.py"):
            text = (BASE_DIR / fname).read_text(encoding="utf-8")
            self.assertNotIn("unsafe_with_html", text, f"Found Streamlit typo in {fname}")
            ast.parse(text, filename=fname)

    def test_all_python_files_parse_cleanly(self) -> None:
        for py_file in Path(BASE_DIR).glob("*.py"):
            ast.parse(py_file.read_text(encoding="utf-8"), filename=py_file.name)


if __name__ == "__main__":
    unittest.main()
