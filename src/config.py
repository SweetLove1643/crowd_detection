# đọc .env (ngưỡng, cooldown, margin...)
import os

from dotenv import load_dotenv

load_dotenv()

MODULE = (os.getenv("MODULE") or "CROWD_DETECTION").upper()
ZONE_ID = os.getenv("ZONE_ID", "51")
ORG_ID = os.getenv("ORG_ID", "103")

MQTT_HOST = os.environ["MQTT_HOST"]
MQTT_PORT = int(os.getenv("MQTT_PORT", "433"))
MQTT_PATH = os.getenv("MQTT_PATH", "/mqtt")
MQTT_USER = os.environ["MQTT_USER"]
MQTT_PASS = os.environ["MQTT_PASS"]
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "phan_van_quan")

STORAGE_URL = os.getenv("STORAGE_URL", "https://aioc.atin.vn/storage/upload/file")

WEIGHT_YOLO_MODEL_PATH = os.getenv("WEIGHT_YOLO_MODEL_PATH", "./resources/runs/train/weights/best.pt")
YOLO_MODEL_CONF = os.getenv("YOLO_MODEL_CONF", "0.6")

NUMBER_PERSON_IS_CROWDED = int(os.getenv("NUMBER_PERSON_IS_CROWDED", "5"))
CROWD_PERSISTENCE_SECONDS = float(os.getenv("CROWD_PERSISTENCE_SECONDS", "4"))
CROWD_BBOX_PADDING_PERCENT = float(os.getenv("CROWD_BBOX_PADDING_PERCENT", "10"))
PUBLISH_ALL_PERSON_BBOX = os.getenv(
    "PUBLISH_ALL_PERSON_BBOX", "false"
).strip().lower() in {"1", "true", "yes", "on"}

if CROWD_PERSISTENCE_SECONDS <= 0:
    raise ValueError("CROWD_PERSISTENCE_SECONDS must be greater than 0")
if CROWD_BBOX_PADDING_PERCENT < 0:
    raise ValueError("CROWD_BBOX_PADDING_PERCENT must not be negative")
