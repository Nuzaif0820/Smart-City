"""Generate a protocol breakdown chart from predicted anomaly CSV outputs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from smart_city_ml import BASE_DIR


def render_anomaly_chart(input_path: Path, output_path: Path, show: bool = False) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: '{input_path}'")

    df = pd.read_csv(input_path, encoding="latin1")
    score_col = next(
        (c for c in ("Anomaly_Score", "Anomaly_IForest", "IForest_Anomaly") if c in df.columns),
        None,
    )
    if score_col is None:
        raise ValueError("No anomaly score column found in CSV.")

    anomalies = df[df[score_col] == -1]
    if anomalies.empty:
        print("ℹ️ No anomalies (-1) present to chart.")
        return

    anomaly_counts = anomalies["Protocol"].value_counts().head(10).reset_index()
    anomaly_counts.columns = ["Protocol", "Count"]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=anomaly_counts,
        x="Count",
        y="Protocol",
        hue="Protocol",
        palette="flare",
        legend=False,
    )
    plt.title("Top 10 Protocols Flagged as Anomalies by the AI", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Flagged Packets", fontsize=12)
    plt.ylabel("Network Protocol Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"💾 Chart saved to '{output_path.name}'.")
    if show:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize anomaly counts by protocol.")
    parser.add_argument("--input", default=str(BASE_DIR / "smart_city_anomalies_predicted_5.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "smart_city_anomalies_chart_5.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    render_anomaly_chart(Path(args.input), Path(args.output), show=args.show)


if __name__ == "__main__":
    main()
