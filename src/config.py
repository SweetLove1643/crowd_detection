# Cấu hình tập trung của hệ thống:
# - Thông tin kết nối môi trường (MQTT, Zone, Storage) được đọc từ file .env
# - Toàn bộ tham số AI, Model, Tracking, Crowd, BBox, Debug được cấu hình trực tiếp tại file này (Single Source of Truth)
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# ==============================================================================
# 1. THÔNG TIN ĐỊNH DANH & KẾT NỐI SERVER (Đọc từ .env)
# ==============================================================================
MODULE = os.environ["MODULE"].strip().upper()
ZONE_ID = os.environ["ZONE_ID"].strip()
ORG_ID = os.environ["ORG_ID"].strip()

MQTT_HOST = os.environ["MQTT_HOST"].strip()
MQTT_PORT = int(os.environ["MQTT_PORT"])
MQTT_PATH = os.environ["MQTT_PATH"].strip()
MQTT_USER = os.environ["MQTT_USER"]
MQTT_PASS = os.environ["MQTT_PASS"]
MQTT_CLIENT_ID = os.environ["MQTT_CLIENT_ID"].strip()
MQTT_TLS_ENABLED = False

STORAGE_URL = os.getenv("STORAGE_URL").strip()

# ==============================================================================
# 2. CẤU HÌNH MODEL AI & INFERENCE (Chỉnh sửa trực tiếp tại đây)
# ==============================================================================
WEIGHT_YOLO_MODEL_PATH = "./resources/runs/train/weights/final.pt"
OPENVINO_MODEL_PATH = "./resources/runs/train/weights/final_openvino_model"
YOLO_INFERENCE_BACKEND = "auto"  # "auto" | "pytorch" | "cuda" | "openvino" | "tensorrt"
YOLO_OPENVINO_DEVICE = "intel:cpu"
YOLO_IMGSZ = 896
YOLO_HALF_PRECISION = True
YOLO_MODEL_CONF = 0.55
YOLO_CPU_THREADS = 6
MAX_CONCURRENT_INFERENCES = 2

# ==============================================================================
# 3. CẤU HÌNH ĐÁNH GIÁ KHOẢNG CÁCH & ĐÁM ĐÔNG (Chỉnh sửa trực tiếp tại đây)
# ==============================================================================
PROXIMITY_ALPHA = 1.5
PROXIMITY_Y_SCALE = 1.5
PROXIMITY_LENGTH_MODE = "max"  # "max" | "height"

NUMBER_PERSON_IS_CROWDED = 5
CROWD_PERSISTENCE_SECONDS = 20.0
CROWD_TRACK_GRACE_SECONDS = 2.5
CROWD_SPATIAL_IOU_THRESHOLD = 0.25
CROWD_CENTER_DISTANCE_RATIO = 0.6
CROWD_EVENT_REPEAT_INTERVAL_SECONDS = 120.0
CROWD_BBOX_PADDING_PERCENT = 10.0

# ==============================================================================
# 4. CẤU HÌNH BBOX REALTIME LÊN WEB VMS (Chỉnh sửa trực tiếp tại đây)
# ==============================================================================
PUBLISH_ALL_PERSON_BBOX = True

# ==============================================================================
# 5. CẤU HÌNH LOG & DEBUG (Chỉnh sửa trực tiếp tại đây)
# ==============================================================================
LOG_LEVEL = "INFO"
DEBUG_VIDEO_ENABLED = False
DEBUG_FRAME_SAVE_ENABLED = False
DEBUG_VIDEO_FPS = 10.0
DEBUG_MAX_SAVED_FRAMES = 200
DEBUG_MAX_SAVED_VIDEOS = 10

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
