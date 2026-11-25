"""Cross-platform CPS infrastructure health and Jetson gateway ping check."""

import argparse
import os
import platform
import subprocess
from datetime import datetime

import psutil

from smart_city_ml import JETSON_SERVER_IP


def ping_host(host: str, timeout_sec: int = 2) -> bool:
    """Ping `host` once across Windows, Linux, and macOS."""
    is_windows = platform.system().lower() == "windows"
    count_flag = "-n" if is_windows else "-c"
    timeout_flag = "-w" if is_windows else "-W"
    timeout_val = str(timeout_sec * 1000) if is_windows else str(timeout_sec)
    try:
        res = subprocess.run(
            ["ping", count_flag, "1", timeout_flag, timeout_val, host],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec + 1,
        )
        return res.returncode == 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Smart City host and lab connectivity diagnostics.")
    parser.add_argument("--host", default=JETSON_SERVER_IP)
    args = parser.parse_args()

    print("🖥️  RUNNING CORE CPS INFRASTRUCTURE HEALTH CHECK 🖥️")
    print("-" * 55)
    print(f"OS Platform       : {platform.system()} {platform.release()}")
    print(f"Current Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cpu_usage = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))

    print("📊 Hardware Resource Metrics:")
    print(f"├── CPU Utilization : {cpu_usage}% ({'ALERT' if cpu_usage > 85 else 'Nominal'})")
    print(f"├── Memory Usage    : {memory.percent}% (Available: {memory.available / (1024**3):.2f} GB)")
    print(f"└── Disk Usage      : {disk.percent}% (Free: {disk.free / (1024**3):.2f} GB)\n")

    print(f"🌐 Network Ping Diagnostics (Target: {args.host}):")
    if ping_host(args.host):
        print("└── ✅ Connection Successful! Lab gateway is responsive.")
    else:
        print("└── ❌ Connection Failed! Gateway unreachable. Check RJ45/subnet connectivity.")
    print("-" * 55)


if __name__ == "__main__":
    main()
