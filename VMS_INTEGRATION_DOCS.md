# TÀI LIỆU CHUẨN GIAO THỨC TÍCH HỢP SMART VMS & HƯỚNG DẪN MIGRATION MINIO S3

> **Dành cho Developer mới:** Tài liệu này tổng hợp toàn bộ giao thức giao tiếp giữa AI Edge Service (`CROWD_DETECTION`) và Smart VMS Core (MQTT WebSocket, MinIO S3 Snapshot, BBox Realtime), đồng thời so sánh chi tiết giữa cơ chế cũ (`AI_MODULES_MQTT.md`) và cơ chế mới nhất (`UPLOAD_ANH_MINIO_S3.md` / `s3_snapshot_url.py`).

---

## 1. SO SÁNH GIAO THỨC CŨ VÀ GIAO THỨC MỚI

| Tiêu chí | Giao thức cũ (`AI_MODULES_MQTT.md`) | Giao thức mới (`UPLOAD_ANH_MINIO_S3.md`) | Vì sao thay đổi? |
| :--- | :--- | :--- | :--- |
| **Giao thức upload ảnh** | **HTTP REST API (multipart/form-data)** qua `POST /storage/upload/file` (port 18649 hoặc domain) | **S3 API trực tiếp (`PutObject`)** vào MinIO S3 (port 18646) dùng `boto3` (AWS SigV4) | API cũ xử lý tuần tự, trần ~1.8 ảnh/s, AI gửi dồn gây bottleneck, timeout, drop event. S3 trực tiếp đạt >15 ảnh/s. |
| **Bên sinh `object_key`** | **Server sinh ngẫu nhiên** và trả về qua response JSON `{"object_key": "..."}` | **AI Edge Client tự sinh trước** theo chuẩn cấu trúc đặt tên | Loại bỏ vòng chờ Server sinh key, giảm độ trễ xử lý. |
| **Định dạng `object_key` (snapshot_url)** | `category/YYYY-MM-DD/camera_<id>/<uuid>.jpg` (do server đặt) | `{category}/{YYYY-MM-DD}/camera_{camera_id}/{uuid}.{ext}` (**Client tự format theo giờ VN UTC+7**) | Thống nhất phân cấp thư mục theo ngày & camera ID để lưu trữ và dọn dẹp (retention). |
| **Quyền truy cập Storage** | Không cần chữ ký AWS, post thẳng HTTP | Dùng `VMS_S3_ACCESS_KEY` & `VMS_S3_SECRET_KEY`, policy `edge-ai-writer` (chỉ có quyền `PutObject`) | Bảo mật cao, client chỉ được ghi vào bucket `vms-captures` với prefix chỉ định, không có quyền xóa/đọc. |
| **Giao thức MQTT & Topic** | MQTT 3.1.1 qua WebSocket (`wss://` hoặc `ws://`) | **GIỮ NGUYÊN KHÔNG ĐỔI** | Không làm ảnh hưởng kiến trúc message broker. |
| **Trường ảnh trong MQTT event** | Gửi `snapshot_url = <object_key>` | **GIỮ NGUYÊN KHÔNG ĐỔI** (`snapshot_url = <object_key>`) | Backend VMS đọc ảnh theo key từ S3 nên payload MQTT giữ nguyên 100%. |

---

## 2. QUY CHUẨN GIAO THỨC TÍCH HỢP CHUẨN SMART VMS (PROD)

Hệ thống AI Edge tương tác với Smart VMS qua **3 kênh chính**:

```
                       ┌───────────────────────── SMART VMS ─────────────────────────┐
                       │                                                             │
  [1] Nhận Camera List │  MQTT Topic: smart_vms/cameras/zone/{ZONE_ID} (retained)    │
  ◄────────────────────┤                                                             │
                       │                                                             │
  [2] Gửi BBox Realtime│  MQTT Topic: smart_vms/ai/bbox/{camera_code} (QoS 0)        │
  ────────────────────►│                                                             │
                       │                                                             │
  [3] Upload Snapshot  │  MinIO S3: PutObject http://{VMS_S3_ENDPOINT}/{BUCKET}/{KEY}│
  ────────────────────►│  (Bucket: vms-captures, Auth: AWS SigV4)                   │
                       │                                                             │
  [4] Gửi Cảnh báo     │  MQTT Topic: smart_vms/ai_events/CROWD_DETECTION (QoS 1)    │
  ────────────────────►│  (kèm snapshot_url = KEY vừa upload ở bước 3)              │
                       └─────────────────────────────────────────────────────────────┘
```

### 2.1. Kênh 1: Nhận danh sách Camera từ VMS
- **Topic subscribe**: `smart_vms/cameras/zone/{ZONE_ID}` (QoS 1)
- **Cơ chế**: VMS publish retained message ngay khi AI Edge kết nối hoặc khi có thay đổi bật/tắt camera trên Web VMS.
- **Payload nhận về**:
  ```json
  {
    "cameras": [
      {
        "id": "6aee1244-632d-43e3-8ef4-447ad539277a",
        "code": "CAM_HN_01",
        "ai_modules": ["CROWD_DETECTION"],
        "rtsp_url": "rtsp://user:pass@192.168.1.x:554/stream",
        "restream_urls": {
          "CROWD_DETECTION": "rtsp://192.168.1.250:18554/live/CAM_HN_01_CROWD"
        }
      }
    ]
  }
  ```
- **Quy tắc xử lý**: Lấy ưu tiên `restream_urls["CROWD_DETECTION"]`, nếu không có fallback về `rtsp_url`. Chỉ chạy camera có module `CROWD_DETECTION`.

### 2.2. Kênh 2: Gửi Bounding Box Realtime (Overlay lên Web VMS)
- **Topic publish**: `smart_vms/ai/bbox/{camera_code}` (QoS 0, gửi liên tục theo từng frame AI inference)
  > *Lưu ý*: Ký tự đặc biệt trong `camera_code` (`/`, `#`, `+`) phải được replace thành `_`.
- **Payload mẫu**:
  ```json
  {
    "camera_code": "CAM_HN_01",
    "camera_id": "6aee1244-632d-43e3-8ef4-447ad539277a",
    "organization_id": 115,
    "ai_modules": "CROWD_DETECTION",
    "timestamp": 1762740000.123,
    "detections": [
      {
        "id": "Người_1",
        "label": "Người",
        "class": "person",
        "confidence": 0.892,
        "bbox": [0.4123, 0.2215, 0.4789, 0.6124],
        "color": "#ff0000",
        "cluster_id": 1,
        "track_id": 1
      },
      {
        "id": "crowd_1",
        "label": "Cụm 1 (5 người)",
        "class": "crowd",
        "bbox": [0.3012, 0.1542, 0.6214, 0.7011],
        "color": "#ff0000",
        "cluster_id": 1,
        "person_count": 5,
        "track_ids": [1, 2, 4, 7, 9]
      }
    ],
    "hud": {
      "text": "Số đám đông: 1 cụm",
      "display_ms": 1000,
      "color": "#ff0000"
    }
  }
  ```
- **Quy tắc**: Tọa độ `bbox` là mảng `[x1, y1, x2, y2]` đã **chuẩn hóa về khoảng 0..1**.

### 2.3. Kênh 3: Upload ảnh Snapshot MinIO S3 (Giao thức mới)
- **Thư viện chuẩn**: `boto3` (Python)
- **Quy tắc đặt Object Key**:
  ```text
  {category}/{YYYY-MM-DD}/camera_{camera_id}/{uuid}.{ext}
  ```
  - `category`: Bắt buộc là **`crowd`** đối với module đám đông (hoặc `anomaly` tùy phân hệ đăng ký trên S3 policy, tài liệu S3 chuẩn định nghĩa `crowd` là chuẩn cho Đám đông).
  - `YYYY-MM-DD`: Ngày theo **giờ Việt Nam (UTC+7)** (`Asia/Ho_Chi_Minh`).
  - `camera_{camera_id}`: Bắt buộc có tiền tố `camera_` + `camera_id`.
  - `{uuid}.{ext}`: `uuid.uuid4().hex + ".jpg"`.
- **Ví dụ Key chuẩn**: `crowd/2026-11-12/camera_6aee1244-632d-43e3-8ef4-447ad539277a/a0782800f25642a8b910123456789abc.jpg`
- **S3 Action**: `s3.put_object(Bucket=VMS_S3_BUCKET, Key=key, Body=image_bytes, ContentType="image/jpeg")`

### 2.4. Kênh 4: Gửi Sự kiện Cảnh báo Đám đông (Crowd Event)
- **Topic publish**: `smart_vms/ai_events/CROWD_DETECTION` (QoS 1)
- **Điều kiện**: Chỉ publish sau khi S3 PutObject thành công HTTP 200.
- **Payload chuẩn theo quy định VMS**:
  ```json
  {
    "ai_modules": "CROWD_DETECTION",
    "submodule_code": "CROWD_DETECTION",
    "camera_id": "6aee1244-632d-43e3-8ef4-447ad539277a",
    "camera_code": "CAM_HN_01",
    "event_time": "2026-11-12T13:45:00+07:00",
    "event_type": "CROWD_DETECTION",
    "entity_type": "PERSON",
    "severity": "HIGH",
    "payload": {
      "type": "HIGH_DENSITY",
      "person_count": 8,
      "threshold": 5,
      "clusters": [
        {
          "cluster_id": 1,
          "person_count": 8,
          "detections": [0, 1, 2, 3, 4, 5, 6, 7],
          "track_ids": [1, 2, 3, 5, 7, 8, 10, 12]
        }
      ]
    },
    "snapshot_url": "crowd/2026-11-12/camera_6aee1244-632d-43e3-8ef4-447ad539277a/a0782800f25642a8b910123456789abc.jpg"
  }
  ```
- **Quy định trường `severity`**:
  - `HIGH`: Cụm có ≥ 10 người.
  - `MEDIUM`: Cụm có 5–9 người.
  - `LOW`: Dưới 5 người (nếu có).

---

## 3. CÁC ĐIỂM CẦN SỬA TRONG SOURCE CODE ĐỂ CẬP NHẬT GIAO THỨC MỚI

Hiện tại, codebase dự án vẫn đang sử dụng module `requests` để POST HTTP multipart lên `STORAGE_URL` (gây lỗi `ConnectionRefusedError` như trong log cũ). Để chuyển sang giao thức MinIO S3 mới, cần thực hiện cập nhật tại **4 vị trí**:

### 3.1. Cài đặt thư viện dependency
- Cần cài thêm: `boto3`
```bash
pip install boto3
```

### 3.2. Cập nhật biến môi trường ([`.env`](.env))
Bỏ hoặc giữ dự phòng biến `STORAGE_URL`, bổ sung 4 biến cấu hình MinIO S3 do VMS cấp:
```ini
# --- CẤU HÌNH MINIO S3 MỚI (Snapshot Upload) ---
VMS_S3_ENDPOINT=http://192.168.1.250:18646
VMS_S3_BUCKET=vms-captures
VMS_S3_ACCESS_KEY=atin
VMS_S3_SECRET_KEY=<secret_key_duoc_cap_rieng>
```

### 3.3. Cập nhật file cấu hình ([`src/config.py`](src/config.py))
Bổ sung các tham số S3 vào `src/config.py`:
```python
# Cấu hình MinIO S3 Upload Snapshot
VMS_S3_ENDPOINT = os.getenv("VMS_S3_ENDPOINT", "http://192.168.1.250:18646").strip()
VMS_S3_BUCKET = os.getenv("VMS_S3_BUCKET", "vms-captures").strip()
VMS_S3_ACCESS_KEY = os.getenv("VMS_S3_ACCESS_KEY", "atin").strip()
VMS_S3_SECRET_KEY = os.getenv("VMS_S3_SECRET_KEY", "").strip()

# Cấu hình Severity theo số người
CROWD_SEVERITY_MEDIUM_MIN_PERSONS = 5
CROWD_SEVERITY_HIGH_MIN_PERSONS = 10
```

### 3.4. Cập nhật module phát tán dữ liệu ([`src/publisher.py`](src/publisher.py))
Thay thế hoàn toàn khối upload `requests.Session()` bằng `boto3.client("s3")`:

1. **Khởi tạo S3 Client Singleton (có connection pool & timeout)**:
   ```python
   import boto3
   from botocore.config import Config

   _s3_client = boto3.client(
       "s3",
       endpoint_url=cf.VMS_S3_ENDPOINT,
       aws_access_key_id=cf.VMS_S3_ACCESS_KEY,
       aws_secret_access_key=cf.VMS_S3_SECRET_KEY,
       region_name="us-east-1",
       config=Config(
           signature_version="s3v4",
           s3={"addressing_style": "path"},
           connect_timeout=5,
           read_timeout=15,
           retries={"max_attempts": 3, "mode": "standard"},
       ),
   )
   ```

2. **Hàm `upload_snapshot` mới chuẩn quy ước S3**:
   ```python
   def upload_snapshot(jpg: bytes, camera_id: str, category: str = "crowd") -> str | None:
       try:
           day = datetime.now(TZ).strftime("%Y-%m-%d")
           key = f"{category}/{day}/camera_{camera_id}/{uuid.uuid4().hex}.jpg"
           _s3_client.put_object(
               Bucket=cf.VMS_S3_BUCKET,
               Key=key,
               Body=jpg,
               ContentType="image/jpeg",
           )
           log.info("S3 Snapshot uploaded camera=%s key=%s", camera_id, key)
           return key
       except Exception:
           log.exception("Upload ảnh MinIO S3 thất bại camera=%s", camera_id)
           return None
   ```

3. **Cập nhật tính mức `severity` linh hoạt trong `publish_crowd_event`**:
   ```python
   def _calc_severity(clusters: list[dict]) -> str:
       max_people = max((c.get("person_count", 0) for c in clusters), default=0)
       if max_people >= cf.CROWD_SEVERITY_HIGH_MIN_PERSONS:
           return "HIGH"
       if max_people >= cf.CROWD_SEVERITY_MEDIUM_MIN_PERSONS:
           return "MEDIUM"
       return "LOW"
   ```
   Gán vào payload event:
   ```python
   "severity": _calc_severity(crowded_clusters)
   ```

---

## 4. CHECKLIST DÀNH CHO DEVELOPER KHI TRIỂN KHAI LÊN PRODUCTION
- [ ] Kiểm tra kết nối từ máy AI Edge tới MinIO:
  `curl -s -o /dev/null -w "%{http_code}\n" http://192.168.1.250:18646/minio/health/live` (phải trả về `200`).
- [ ] Đảm bảo `VMS_S3_SECRET_KEY` được cấu hình qua `.env`, không hardcode vào git repo.
- [ ] Đồng bộ thời gian hệ thống NTP trên máy AI Edge (lệch quá 15 phút MinIO sẽ báo lỗi `RequestTimeTooSkewed`).
- [ ] Kiểm tra sau khi event bắn lên, mở link ảnh trên backend VMS:
  `http://192.168.1.250:18640/api/v1/snapshot/image?object_key=<object_key>` để đảm bảo ảnh load thành công HTTP 200.
