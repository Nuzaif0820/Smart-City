"""Direct MQTT actuator script for smart streetlights and traffic signals."""

import argparse
import time

from smart_city_ml import JETSON_SERVER_IP, send_mqtt_command

STREETLIGHT_TOPIC = "city/lighting/control"
TRAFFIC_LIGHT_TOPIC = "city/traffic/control"


def main() -> None:
    parser = argparse.ArgumentParser(description="Operate Smart City lighting and traffic MQTT endpoints.")
    parser.add_argument("--broker", default=JETSON_SERVER_IP)
    parser.add_argument("--light", default="ON", choices=["ON", "OFF"])
    parser.add_argument("--signal", default="RED", choices=["RED", "YELLOW", "GREEN"])
    args = parser.parse_args()

    print(f"🔌 Sending hardware MQTT commands to {args.broker}:1883...")
    ok1, msg1 = send_mqtt_command(STREETLIGHT_TOPIC, args.light, broker_ip=args.broker)
    print(f"💡 Street Lights -> {args.light}: {'✅' if ok1 else '❌'} {msg1}")

    time.sleep(0.5)

    ok2, msg2 = send_mqtt_command(TRAFFIC_LIGHT_TOPIC, args.signal, broker_ip=args.broker)
    print(f"🚦 Traffic Signal -> {args.signal}: {'✅' if ok2 else '❌'} {msg2}")


if __name__ == "__main__":
    main()
