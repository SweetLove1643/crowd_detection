# khởi động, nối các thành phần
import logging

import src.config as config
from src.setup_logging import setup_logging, log_startup_banner
from src.mqtt_client import build_client, stop_camera_workers

if __name__ == "__main__":
    setup_logging()
    log_startup_banner()
    client = build_client()
    try:
        client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
        client.loop_forever()
    except KeyboardInterrupt:
        logging.getLogger("main").info("Nhận Ctrl+C, đang dừng workers...")
    finally:
        try:
            client.disconnect()
        finally:
            stop_camera_workers()
