# Cấu hình tập trung của hệ thống:
# - Thông tin kết nối môi trường (MQTT, Zone, S3) được đọc từ file .env
# - Toàn bộ tham số AI, FFmpeg, Tracking, Crowd, BBox, Debug chỉnh sửa trực tiếp tại đây (Single Source of Truth)
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

# Cấu hình MinIO S3 Snapshot Upload
VMS_S3_ENDPOINT = os.getenv("VMS_S3_ENDPOINT", "http://192.168.1.250:18646").strip()
VMS_S3_BUCKET = os.getenv("VMS_S3_BUCKET", "vms-captures").strip()                  
VMS_S3_ACCESS_KEY = os.getenv("VMS_S3_ACCESS_KEY", "atin").strip()                
VMS_S3_SECRET_KEY = os.getenv("VMS_S3_SECRET_KEY", "").strip()              

# ==============================================================================
# 2. CẤU HÌNH MODEL AI, FFmpeg & INFERENCE (Chỉnh sửa trực tiếp tại đây)
# ==============================================================================
WEIGHT_YOLO_MODEL_PATH = "./resources/runs/train/weights/24_nov.pt"         # Đường dẫn weights YOLO (.pt hoặc .engine); nếu dùng "tensorrt" mà thiếu .engine sẽ tự export từ .pt cùng tên
OPENVINO_MODEL_PATH = "./resources/runs/train/weights/final_openvino_model"
YOLO_INFERENCE_BACKEND = "tensorrt"   # Runtime suy luận: "auto" | "pytorch" | "cuda" | "openvino" | "tensorrt" 
YOLO_OPENVINO_DEVICE = "intel:cpu" # Thiết bị OpenVINO ("intel:cpu", "intel:gpu", "auto")
YOLO_IMGSZ = 1280                
YOLO_HALF_PRECISION = True        # Bật FP16 (nhanh hơn, ít VRAM hơn) khi chạy trên GPU CUDA/TensorRT
YOLO_MODEL_CONF = 0.7            
YOLO_MODEL_IOU = 0.55              # Ngưỡng IoU khi lọc NMS; giảm xuống (0.5-0.6) nếu 1 người bị tách thành 2 bbox
YOLO_CPU_THREADS = 6              # Số luồng CPU cấp cho PyTorch (áp dụng cho cả CPU lẫn GPU); nên <= số nhân vật lý trừ phần cho FFmpeg
MAX_CONCURRENT_INFERENCES = 1     # Số camera gọi model.predict() đồng thời; để 1 tránh tranh chấp lock GPU (ByteTrack chạy riêng ngoài lock)
FFMPEG_MAX_WIDTH = 1280           # Giới hạn chiều rộng tối đa đầu ra FFmpeg (giữ tỷ lệ gốc, giảm tải pipe/RAM/resize; 0 = giữ nguyên gốc)

# ==============================================================================
# 3. CẤU HÌNH ĐÁNH GIÁ KHOẢNG CÁCH & ĐÁM ĐÔNG (Chỉnh sửa trực tiếp tại đây)
# ==============================================================================
PROXIMITY_ALPHA = 2.0             # Hệ số khoảng cách đứng gần (alpha * chiều dài cơ thể TB); tăng = dễ gom thành cụm hơn
PROXIMITY_Y_SCALE = 1.2           # Hệ số kéo giãn trục dọc Y bù góc nghiêng camera; camera nhìn chéo mạnh tăng lên, nhìn thẳng ~1.0
PROXIMITY_LENGTH_MODE = "max"     # max | h

NUMBER_PERSON_IS_CROWDED = 5     
CROWD_PERSISTENCE_SECONDS = 30.0  
CROWD_TRACK_GRACE_SECONDS = 3.5   # Thời gian chịu mất dấu tạm thời (giây) trước khi reset bộ đếm CROWD_PERSISTENCE_SECONDS
CROWD_SPATIAL_IOU_THRESHOLD = 0.25 # Độ chồng lấp IoU tối thiểu giữa 2 frame để khớp là cùng 1 cụm đám đông
CROWD_CENTER_DISTANCE_RATIO = 0.8  # Khoảng cách tâm tương đối tối đa để vẫn coi là cùng 1 cụm đám đông
CROWD_EVENT_REPEAT_INTERVAL_SECONDS = 180.0 # Chu kỳ (giây) gửi lặp lại sự kiện lên VMS nếu đám đông vẫn còn
CROWD_BBOX_PADDING_PERCENT = 10.0  # Phần trăm nới rộng 4 cạnh bbox đỏ bao quanh cụm đám đông

CROWD_SEVERITY_MEDIUM_MIN_PERSONS = 5 # Số người >= ngưỡng này đánh mức cảnh báo "MEDIUM"
CROWD_SEVERITY_HIGH_MIN_PERSONS = 10  # Số người >= ngưỡng này đánh mức cảnh báo "HIGH"

# ==============================================================================
# 4. CẤU HÌNH BBOX REALTIME LÊN WEB VMS (Chỉnh sửa trực tiếp tại đây)
# ==============================================================================
PUBLISH_ALL_PERSON_BBOX = True    # True = gửi thêm bbox xanh cho người đứng ngoài đám đông; False = chỉ gửi khi có đám đông
CROWD_BBOX_HOLD_SECONDS = 3.0     # Thời gian giữ ổn định bbox đỏ (giây) khi người ở rìa tạm tách ra ngoài (chống giật đỏ/xanh)
CROWD_BBOX_SMOOTHING_ALPHA = 0.35 # Hệ số làm mượt EMA tọa độ bbox đỏ (0 < alpha <= 1.0; nhỏ = mượt hơn, 1.0 = tắt làm mượt)

# ==============================================================================
# 5. CẤU HÌNH LOG & DEBUG (Chỉnh sửa trực tiếp tại đây)
# ==============================================================================
LOG_LEVEL = "INFO"                # Mức log ghi ra console và file ("DEBUG" | "INFO" | "WARNING" | "ERROR")
DEBUG_VIDEO_ENABLED = False       
DEBUG_FRAME_SAVE_ENABLED = False 
DEBUG_VIDEO_FPS = 10.0         
DEBUG_MAX_SAVED_FRAMES = 500    
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
if FFMPEG_MAX_WIDTH < 0:
    raise ValueError("FFMPEG_MAX_WIDTH must not be negative")
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
if not 0 < CROWD_SEVERITY_MEDIUM_MIN_PERSONS < CROWD_SEVERITY_HIGH_MIN_PERSONS:
    raise ValueError("Require 0 < CROWD_SEVERITY_MEDIUM_MIN_PERSONS < CROWD_SEVERITY_HIGH_MIN_PERSONS")
if CROWD_BBOX_HOLD_SECONDS < 0:
    raise ValueError("CROWD_BBOX_HOLD_SECONDS must not be negative")
if not 0.0 < CROWD_BBOX_SMOOTHING_ALPHA <= 1.0:
    raise ValueError("CROWD_BBOX_SMOOTHING_ALPHA must be in (0, 1]")
if YOLO_INFERENCE_BACKEND.strip().lower() not in {"auto", "pytorch", "cuda", "openvino", "tensorrt"}:
    raise ValueError("YOLO_INFERENCE_BACKEND must be one of: auto, pytorch, cuda, openvino, tensorrt")
