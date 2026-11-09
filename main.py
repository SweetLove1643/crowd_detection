# khởi động, nối các thành phần
import src.config as config
from src.setup_logging import setup_logging, log_startup_banner
from src.mqtt_client import build_client

if __name__ == "__main__":
    setup_logging()
    log_startup_banner()
    client = build_client()
    client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=30)
    client.loop_forever()
