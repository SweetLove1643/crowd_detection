# đọc .env (ngưỡng, cooldown, margin...)
import os

from dotenv import load_dotenv

load_dotenv()

# Ép FFmpeg đọc RTSP qua TCP (tránh rớt gói UDP gây lỗi HEVC POC/RPS)
# và tắt log rác không nghiêm trọng từ bộ giải mã C của FFmpeg (AV_LOG_FATAL = 8)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


MODULE = (os.getenv("MODULE") or "CROWD_DETECTION").strip().upper()
ZONE_ID = os.getenv("ZONE_ID", "51").strip()
ORG_ID = os.getenv("ORG_ID", "103").strip()

MQTT_HOST = os.environ["MQTT_HOST"].strip()
MQTT_PORT = int(os.getenv("MQTT_PORT", "443"))
MQTT_PATH = os.getenv("MQTT_PATH", "/mqtt").strip()
MQTT_USER = os.environ["MQTT_USER"]
MQTT_PASS = os.environ["MQTT_PASS"]
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "phan_van_quan").strip()
MQTT_TLS_ENABLED = _env_bool(
    "MQTT_TLS_ENABLED",
    "true" if MQTT_PORT == 443 else "false",
)

STORAGE_URL = os.getenv(
    "STORAGE_URL",
    "https://aioc.atin.vn/storage/upload/file",
).strip()

WEIGHT_YOLO_MODEL_PATH = os.getenv(
    "WEIGHT_YOLO_MODEL_PATH",
    "./resources/runs/train/weights/best.pt",
).strip()
OPENVINO_MODEL_PATH = os.getenv(
    "OPENVINO_MODEL_PATH",
    "./resources/runs/train/weights/best_openvino_model",
).strip()
YOLO_INFERENCE_BACKEND = (
    os.getenv("YOLO_INFERENCE_BACKEND", "auto").strip().lower()
)
YOLO_OPENVINO_DEVICE = os.getenv("YOLO_OPENVINO_DEVICE", "intel:cpu").strip()
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
YOLO_HALF_PRECISION = _env_bool("YOLO_HALF_PRECISION", "true")
YOLO_MODEL_CONF = float(os.getenv("YOLO_MODEL_CONF", "0.5"))
YOLO_CPU_THREADS = int(os.getenv("YOLO_CPU_THREADS", "4"))
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))

PROXIMITY_ALPHA = float(os.getenv("PROXIMITY_ALPHA", "1.5"))
PROXIMITY_Y_SCALE = float(os.getenv("PROXIMITY_Y_SCALE", "1.5"))
PROXIMITY_LENGTH_MODE = (
    os.getenv("PROXIMITY_LENGTH_MODE", "max").strip().lower()
)

DEBUG_VIDEO_ENABLED = _env_bool("DEBUG_VIDEO_ENABLED", "false")
DEBUG_FRAME_SAVE_ENABLED = _env_bool("DEBUG_FRAME_SAVE_ENABLED", "false")
DEBUG_VIDEO_FPS = float(os.getenv("DEBUG_VIDEO_FPS", "10.0"))
DEBUG_MAX_SAVED_FRAMES = int(os.getenv("DEBUG_MAX_SAVED_FRAMES", "200"))
DEBUG_MAX_SAVED_VIDEOS = int(os.getenv("DEBUG_MAX_SAVED_VIDEOS", "10"))

NUMBER_PERSON_IS_CROWDED = int(os.getenv("NUMBER_PERSON_IS_CROWDED", "5"))
CROWD_PERSISTENCE_SECONDS = float(os.getenv("CROWD_PERSISTENCE_SECONDS", "4"))
CROWD_TRACK_GRACE_SECONDS = float(
    os.getenv("CROWD_TRACK_GRACE_SECONDS", "2.5")
)
CROWD_SPATIAL_IOU_THRESHOLD = float(
    os.getenv("CROWD_SPATIAL_IOU_THRESHOLD", "0.25")
)
CROWD_CENTER_DISTANCE_RATIO = float(
    os.getenv("CROWD_CENTER_DISTANCE_RATIO", "0.6")
)
CROWD_EVENT_REPEAT_INTERVAL_SECONDS = float(
    os.getenv("CROWD_EVENT_REPEAT_INTERVAL_SECONDS", "10")
)
CROWD_BBOX_PADDING_PERCENT = float(os.getenv("CROWD_BBOX_PADDING_PERCENT", "10"))
PUBLISH_ALL_PERSON_BBOX = _env_bool("PUBLISH_ALL_PERSON_BBOX", "false")
BBOX_HEARTBEAT_INTERVAL_SECONDS = float(
    os.getenv("BBOX_HEARTBEAT_INTERVAL_SECONDS", "0.2")
)
BBOX_MAX_STALE_SECONDS = float(os.getenv("BBOX_MAX_STALE_SECONDS", "1.8"))

if CROWD_PERSISTENCE_SECONDS <= 0:
    raise ValueError("CROWD_PERSISTENCE_SECONDS must be greater than 0")
if CROWD_TRACK_GRACE_SECONDS <= 0:
    raise ValueError("CROWD_TRACK_GRACE_SECONDS must be greater than 0")
if CROWD_EVENT_REPEAT_INTERVAL_SECONDS <= 0:
    raise ValueError("CROWD_EVENT_REPEAT_INTERVAL_SECONDS must be greater than 0")
if CROWD_BBOX_PADDING_PERCENT < 0:
    raise ValueError("CROWD_BBOX_PADDING_PERCENT must not be negative")
if YOLO_CPU_THREADS <= 0:
    raise ValueError("YOLO_CPU_THREADS must be greater than 0")
if MAX_CONCURRENT_INFERENCES <= 0:
    raise ValueError("MAX_CONCURRENT_INFERENCES must be greater than 0")
if YOLO_IMGSZ <= 0:
    raise ValueError("YOLO_IMGSZ must be greater than 0")
if DEBUG_VIDEO_FPS <= 0:
    raise ValueError("DEBUG_VIDEO_FPS must be greater than 0")
if PROXIMITY_ALPHA <= 0:
    raise ValueError("PROXIMITY_ALPHA must be greater than 0")
if PROXIMITY_Y_SCALE <= 0:
    raise ValueError("PROXIMITY_Y_SCALE must be greater than 0")
if PROXIMITY_LENGTH_MODE not in {"max", "height"}:
    raise ValueError("PROXIMITY_LENGTH_MODE must be 'max' or 'height'")
