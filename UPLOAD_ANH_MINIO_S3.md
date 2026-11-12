---
title: "Upload ảnh snapshot trực tiếp lên MinIO S3 — PROD"
subtitle: "Thay thế Storage API (/storage/upload/file, :18649) — áp dụng cho AI Edge và Backend"
date: "2026-11-12"
---

# 0. Tóm tắt thay đổi

Storage API cũ (`minio-api`, `POST /upload/file`) xử lý tuần tự, trần khoảng 1,8 ảnh/giây, trong khi AI Edge gửi khoảng 5 ảnh/giây → hàng đợi dồn, timeout. Từ 2026-11-12:

| Thành phần | Trước | Sau |
|---|---|---|
| AI Edge | `POST http://192.168.1.250:18649/upload/file` (multipart) | S3 `PutObject` thẳng vào MinIO `http://192.168.1.250:18646` |
| Backend | Gọi Storage API, fallback S3 | Chỉ dùng S3 qua `services/object_storage.py` (đã bỏ Storage API) |
| `object_key` | Storage API sinh và trả trong response | Client tự đặt theo quy ước §3.1 |
| Payload MQTT | `snapshot_url = <object_key>` | **Không đổi** |
| Xem ảnh | `/api/v1/snapshot/image?object_key=` | **Không đổi** |

Storage API vẫn chạy song song để rollback, sẽ gỡ sau 72 giờ không còn traffic.

# 1. Thông số kết nối

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Endpoint S3 | `http://192.168.1.250:18646` | MinIO PROD, mạng LAN/VPN |
| Bucket | `vms-captures` | Private |
| Region | `us-east-1` | |
| Addressing | path-style | `http://host:port/{bucket}/{key}` |
| Chữ ký | AWS Signature V4 | |
| Access key | `atin` | Policy `edge-ai-writer` |
| Secret key | cấp riêng qua kênh bảo mật | Không ghi vào code, repo, log |

Quyền của user `atin`: **chỉ `PutObject`** vào 16 prefix ở §3.2. Không đọc, không liệt kê, không xoá được — đúng thiết kế.

Khuyến nghị cấu hình phía Edge qua ENV:

| ENV | Giá trị |
|---|---|
| `VMS_S3_ENDPOINT` | `http://192.168.1.250:18646` |
| `VMS_S3_BUCKET` | `vms-captures` |
| `VMS_S3_ACCESS_KEY` | `atin` |
| `VMS_S3_SECRET_KEY` | (secret được cấp) |

Kiểm tra kết nối từ máy Edge:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://192.168.1.250:18646/minio/health/live   # 200
```

# 2. Luồng gửi ảnh

Ảnh không gửi qua MQTT. Mỗi event có ảnh đi theo 4 bước:

| Bước | Việc | Kết quả |
|---|---|---|
| 1 | Đặt `object_key` theo quy ước §3.1 | Ví dụ `zone/2026-11-12/camera_<camera_id>/<uuid>.jpg` |
| 2 | `PutObject` ảnh vào bucket `vms-captures` với key đó | MinIO trả HTTP 200 |
| 3 | Publish event MQTT với `"snapshot_url": "<object_key>"` | Backend lưu event kèm ảnh |
| 4 | Kiểm tra ảnh trên VMS | Ảnh hiển thị qua backend |

Chỉ publish MQTT **sau khi** bước 2 trả 200.

# 3. Bước 1 + 2 — Upload ảnh

## 3.1 Quy ước `object_key`

```
{category}/{YYYY-MM-DD}/camera_{camera_id}/{uuid}.{ext}
```

| Thành phần | Quy định |
|---|---|
| `category` | Một trong 16 giá trị ở §3.2 (sai → 403 `AccessDenied`) |
| `YYYY-MM-DD` | Ngày chụp theo **giờ Việt Nam (UTC+7)**, không theo UTC |
| `camera_{camera_id}` | Tiền tố `camera_` + UUID (hoặc mã) camera, giống form `camera_id` cũ |
| `{uuid}.{ext}` | Tên duy nhất, ví dụ `uuid4().hex + ".jpg"`; chỉ dùng `[A-Za-z0-9._-]` |
| Header `Content-Type` | `image/jpeg` hoặc `image/png` |
| Kích thước | Khuyến nghị ≤ 10 MB (giữ như giới hạn cũ) |

Sai ví dụ: `fire/test-cam-id/45fed7a3….jpg` — thiếu thư mục ngày và tiền tố `camera_`; ảnh vẫn lưu nhưng không theo chuẩn, khó tra cứu và dọn dữ liệu theo ngày.

## 3.2 Danh sách `category` được phép

| `category` | Module |
|---|---|
| `anomaly` | Hành vi bất thường (ANOMALY) |
| `auda` | AUDA |
| `crowd` | Đám đông |
| `electric` | Điện |
| `events` | Sự kiện chung |
| `face` | Khuôn mặt (FACE) |
| `fire` | Cháy |
| `footfall` | Đếm lượt |
| `person` | Người |
| `plate` | Biển số (PLATE) |
| `ppe` | Bảo hộ lao động |
| `retail` | Bán lẻ |
| `smoke` | Khói |
| `thermal` | Nhiệt (THERMAL) |
| `vehicle` | Phương tiện |
| `zone` | Vùng (ZONE) |

Cần thêm category mới: báo vận hành bổ sung vào policy `edge-ai-writer`.

## 3.3 Ví dụ Python (boto3)

```python
import os, uuid, datetime, boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["VMS_S3_ENDPOINT"],
    aws_access_key_id=os.environ["VMS_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["VMS_S3_SECRET_KEY"],
    region_name="us-east-1",
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                  connect_timeout=5, read_timeout=15,
                  retries={"max_attempts": 3, "mode": "standard"}),
)  # tạo MỘT lần khi khởi động, dùng lại cho mọi ảnh

VN_TZ = datetime.timezone(datetime.timedelta(hours=7))

def upload(category: str, camera_id: str, jpeg: bytes) -> str:
    day = datetime.datetime.now(VN_TZ).strftime("%Y-%m-%d")
    key = f"{category}/{day}/camera_{camera_id}/{uuid.uuid4().hex}.jpg"
    s3.put_object(Bucket=os.environ.get("VMS_S3_BUCKET", "vms-captures"),
                  Key=key, Body=jpeg, ContentType="image/jpeg")
    return key  # dùng làm snapshot_url của event
```

## 3.4 Ví dụ curl (≥ 7.75, có `--aws-sigv4`)

```bash
KEY="zone/$(TZ=Asia/Ho_Chi_Minh date +%F)/camera_6aee1244-632d-43e3-8ef4-447ad539277a/$(cat /proc/sys/kernel/random/uuid | tr -d -).jpg"
curl --fail --aws-sigv4 "aws:amz:us-east-1:s3" \
  --user "$VMS_S3_ACCESS_KEY:$VMS_S3_SECRET_KEY" \
  -H "Content-Type: image/jpeg" -T snapshot.jpg \
  "$VMS_S3_ENDPOINT/vms-captures/$KEY" && echo "$KEY"
```

## 3.5 Ví dụ MinIO Client (`mc`)

```bash
mc alias set vms "$VMS_S3_ENDPOINT" "$VMS_S3_ACCESS_KEY" "$VMS_S3_SECRET_KEY"
mc cp --attr "Content-Type=image/jpeg" snapshot.jpg \
  "vms/vms-captures/fire/$(TZ=Asia/Ho_Chi_Minh date +%F)/camera_CAM01/$(cat /proc/sys/kernel/random/uuid | tr -d -).jpg"
```

## 3.6 So sánh với Storage API cũ

| Storage API cũ (form-data) | S3 mới |
|---|---|
| `file` | Body của `PutObject` |
| `camera_id` | Phần `camera_{camera_id}` trong key |
| `category` | Phần đầu của key |
| Response `object_key` | Client tự đặt trước khi upload |
| Response `url` (presigned nội bộ) | Không có; xem ảnh qua backend §5 |

# 4. Bước 3 — Publish event kèm ảnh

Đặt `object_key` vào trường ảnh của payload MQTT:

| Module | Trường chứa `object_key` |
|---|---|
| Mặc định (ZONE, ANOMALY, FIRE, …) | `snapshot_url` |
| FACE | `image` |
| THERMAL | `frame_url` |
| PLATE (2 ảnh) | Toàn cảnh → `snapshot_url`; crop biển số → `plate_crop_url` / `snapshot_base64` / `payload.vehicle.license_plate.crop_url` |

Ví dụ event ZONE trên topic `smart_vms/ai_events/ZONE`:

```json
{
  "ai_modules": "ZONE",
  "camera_id": "6aee1244-632d-43e3-8ef4-447ad539277a",
  "event_time": "2026-11-12T08:30:00+07:00",
  "payload": { "zone_id": "<ai_zone uuid>" },
  "snapshot_url": "zone/2026-11-12/camera_6aee1244-632d-43e3-8ef4-447ad539277a/a0782800f256.jpg"
}
```

Chi tiết payload từng module: `TAI_LIEU_CHUAN_PAYLOAD_MQTT_AI_PROD.md`.

# 5. Bước 4 — Kiểm tra ảnh

Mở `http://192.168.1.250:18640/api/v1/snapshot/image?object_key=<object_key>` → trả về ảnh (`200 image/jpeg`); key không tồn tại → `404`.

User `atin` không đọc được ảnh; muốn xem phải qua backend như trên.

# 6. File mẫu

`backend/docs/samples/s3_snapshot_url.py` — một file, một hàm: upload ảnh lên S3, trả `snapshot_url`.

```bash
pip install boto3
python s3_snapshot_url.py vehicle <uuid camera> scene.jpg
# → vehicle/2026-11-12/camera_<uuid camera>/<uuid>.jpg   (đặt vào "snapshot_url")
```

Trong code Edge:

```python
from s3_snapshot_url import upload_snapshot

event["snapshot_url"] = upload_snapshot("vehicle", camera_id, jpeg_bytes)
# PLATE: thêm ảnh crop biển số
event["plate_crop_url"] = upload_snapshot("plate", camera_id, crop_bytes)
```

# 7. Mã lỗi

| HTTP | S3 code | Ý nghĩa | Xử lý phía Edge |
|---|---|---|---|
| 200 | — | Thành công | Publish MQTT với key |
| 403 | `AccessDenied` | Sai prefix/category, hoặc key bị thu hồi | Không retry; báo vận hành |
| 403 | `SignatureDoesNotMatch` | Sai secret | Không retry; kiểm tra secret |
| 403 | `RequestTimeTooSkewed` | Giờ máy Edge lệch > 15 phút | Đồng bộ NTP |
| 404 | `NoSuchBucket` | Sai tên bucket | Sửa cấu hình |
| 5xx / timeout | — | MinIO tạm lỗi/quá tải | Retry có backoff (tối đa 3 lần); hết retry thì publish event không kèm ảnh |

# 8. Yêu cầu tích hợp phía Edge

1. Tạo S3 client một lần khi khởi động, dùng lại (pool kết nối); không tạo client cho mỗi ảnh.
2. Upload trong hàng đợi/worker riêng (4–8 luồng), không chặn vòng lặp inference.
3. Timeout: connect 5 s, read 15 s. Retry 5xx/timeout có backoff; không retry 403.
4. Chỉ publish MQTT sau khi `PutObject` trả 200.
5. Đồng bộ giờ NTP.
6. Ngày trong key theo giờ Việt Nam (UTC+7).
7. Không ghi secret vào log, ảnh chụp màn hình, repo.

# 9. Chuyển đổi và rollback

| Bước | Việc |
|---|---|
| 1 | Kiểm tra kết nối `health/live` từ máy Edge (§1) |
| 2 | Upload thử 1 ảnh `category=fire`, gửi `object_key` cho vận hành kiểm tra hiển thị (§5) |
| 3 | Bật S3 cho 1 camera, theo dõi 30 phút |
| 4 | Bật toàn bộ camera |
| 5 | Giữ code gọi Storage API làm fallback (cờ ENV) 72 giờ, sau đó gỡ |

Rollback: tắt cờ ENV → quay lại `POST http://192.168.1.250:18649/upload/file`. Storage API vẫn chạy tới khi vận hành xác nhận không còn traffic 72 giờ.

# 10. Hiệu năng (load test 2026-11-12)

Ảnh 250 KB, gửi theo nhịp cố định (open-loop), trong lúc disk MinIO đang bận dọn dữ liệu cũ:

| Đường | Tải | Thành công | p50 | p95 | Thông lượng |
|---|---|---|---|---|---|
| S3 trực tiếp | 5 ảnh/s | 300/300 | 372 ms | 3,1 s | 4,9/s |
| Storage API cũ | 5 ảnh/s | 198/300 (102 timeout) | 23,7 s | 35,9 s | 1,8/s |
| S3 trực tiếp | 15 ảnh/s | 450/450 | 783 ms | 5,3 s | 14,9/s |
| Storage API cũ | 15 ảnh/s | 0/450 (timeout) | — | — | 0 |

Nguyên nhân Storage API nghẽn: handler `async def upload_file` gọi hàm ghi MinIO đồng bộ → chặn event loop, xử lý từng request một.

# 11. Backend (nội bộ) — upload mới

## 11.1 Module dùng chung

Mọi chỗ backend ghi/đọc ảnh đi qua `backend/services/object_storage.py`:

| Hàm | Việc |
|---|---|
| `get_client()` | boto3 S3 client dùng chung (SigV4, path-style, pool 20, timeout 5/15 s, retry 2) |
| `build_object_key(category, owner_id, filename)` | Sinh key `{category}/{YYYY-MM-DD giờ VN}/camera_{owner_id}/{uuid}.{ext}` |
| `upload_bytes(category, owner_id, filename, content, content_type)` | Sinh key + `PutObject`, trả `object_key` |
| `put_bytes(key, content, content_type)` | `PutObject` với key cho sẵn (chạy trong thread, không chặn event loop) |
| `presign_get(key, expires)` | Ký URL GET tại chỗ (không gọi mạng) |

Lỗi: `ObjectStorageUnavailable` (thiếu cấu hình) và `ObjectStorageError` (MinIO lỗi).

## 11.2 Cấu hình (ENV, `runtime/backend.env`)

| ENV | Ý nghĩa |
|---|---|
| `CAPTURE_S3_ENDPOINT` | Endpoint MinIO nội bộ (PROD: `http://localhost:18646`) |
| `CAPTURE_S3_ACCESS_KEY` / `CAPTURE_S3_SECRET_KEY` | Tài khoản backend (không còn giá trị mặc định trong code) |
| `CAPTURE_S3_BUCKET` | `vms-captures` |
| `CAPTURE_S3_REGION` | `us-east-1` |
| `CAPTURE_S3_PUBLIC_ENDPOINT` | Tuỳ chọn — host dùng khi ký URL presign cho trình duyệt |

Thiếu endpoint/key: backend vẫn khởi động, in cảnh báo `[ObjectStorage]`; upload ảnh sản phẩm bán lẻ trả 503, các chức năng khác bỏ qua ảnh theo bảng §11.3.

`STORAGE_API_BASE_URL` đã bỏ — backend không còn gọi Storage API.

## 11.3 Các chức năng upload của backend

| Chức năng | `category` | Thư mục camera | Lỗi trả về |
|---|---|---|---|
| Ảnh khuôn mặt nhân viên (Smart Face) | `smart_face_employee` | `camera_{camera_id}` hoặc `camera_EMP-{employee_id}` | Trả `None`, không chặn luồng |
| Logo công ty | `company_logo` | `camera_COMPANY-{organization_id}` | Trả `None` |
| Ảnh đại diện công ty | `company_avatar` | `camera_COMPANY-{organization_id}` | Trả `None` |
| Ảnh sản phẩm bán lẻ (heatmap) | `retail_product` | `camera_RETAIL-PRODUCT-{organization_id}` | 503 thiếu cấu hình / 502 MinIO lỗi |
| Âm thanh cảnh báo (bản sao) | `alert_sound` | `camera_ALERT-SOUND-{organization_id}` | Best-effort, bỏ qua lỗi |
| Sơ đồ mặt bằng | key riêng `floorplans/{type}_{node}_{token}{ext}` | — | Lỗi → lưu file tĩnh local |

## 11.4 Đọc ảnh

`GET /api/v1/snapshot/image?object_key=<key>` (cũng nhận URL MinIO đầy đủ): backend đọc thẳng từ MinIO bằng `object_storage.get_client()`. Key không tồn tại → 404. Đã bỏ bước gọi `/files/url` của Storage API.

## 11.5 Vận hành

| Việc | Lệnh |
|---|---|
| Tạo/cập nhật user Edge + kiểm tra quyền | `bash .agents/scripts/minio_provision_users.sh --verify` |
| Theo dõi traffic Storage API còn lại | `docker logs --since 5m aioc_prod_minio_api 2>&1 \| grep -c 'upload/file'` (mục tiêu 0) |
| Log backend | `docker compose -f staging-compose.yml logs --tail=200 backend` |
