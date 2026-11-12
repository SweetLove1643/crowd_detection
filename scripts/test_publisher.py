"""Script kiểm tra gửi thử Crowd Event lên VMS theo giao thức MinIO S3 & MQTT WebSocket mới.

Quy trình hoạt động:
    1. Tạo 1 ảnh snapshot JPEG mô phỏng đám đông.
    2. Upload ảnh lên MinIO S3 qua upload_snapshot() -> Nhận về object_key.
    3. Dùng object_key đó làm snapshot_url để đóng gói payload Crowd Event (kèm tính severity theo số người).
    4. Publish Crowd Event (QoS 1) lên MQTT topic: smart_vms/ai_events/CROWD_DETECTION và chờ Broker xác nhận (ACK).
    5. In đầy đủ JSON Payload và link xem ảnh để đối chiếu trên VMS.

Cách chạy:
    python scripts/test_publisher.py [--camera-id <UUID>] [--camera-code <CODE>] [--people <COUNT>]

Ví dụ:
    python scripts/test_publisher.py --people 6   # Test mức MEDIUM
    python scripts/test_publisher.py --people 12  # Test mức HIGH
"""

import argparse
import json
import logging
import ssl
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import module src
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from src import config as cf
from src.publisher import TZ, _crowd_severity, upload_snapshot

# Cấu hình logging ra console rõ ràng
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_publisher")


def create_dummy_snapshot(text: str = "TEST CROWD EVENT") -> bytes:
    """Tạo một ảnh JPEG mẫu (1280x720) có vẽ text thông tin test."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y in range(720):
        frame[y, :, :] = (int(30 + y * 0.05), int(40 + y * 0.05), int(60 + y * 0.05))

    cv2.putText(
        frame,
        text,
        (80, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        (80, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Module: {cf.MODULE} | Endpoint: {cf.VMS_S3_ENDPOINT}",
        (80, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(frame, (300, 400), (900, 650), (0, 0, 255), 3)
    cv2.putText(
        frame,
        "Simulated Crowd Cluster",
        (310, 385),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Không encode được ảnh JPEG test")
    return buffer.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test publish crowd event to VMS")
    parser.add_argument(
        "--camera-id",
        default="e53b2945-fae2-448c-ad78-ab45ecbe68ad",
        help="Camera ID test (UUID hoặc mã)",
    )
    parser.add_argument(
        "--camera-code",
        default="loi-vao-nha-tron",
        help="Camera Code hiển thị trên VMS",
    )
    parser.add_argument(
        "--people",
        type=int,
        default=6,
        help="Số lượng người trong cụm (>=5: MEDIUM, >=10: HIGH)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU KIỂM TRA: UPLOAD SNAPSHOT S3 -> TẠO & GỬI CROWD EVENT LÊN VMS")
    print("=" * 70)
    print(f"📌 MQTT Broker   : {cf.MQTT_HOST}:{cf.MQTT_PORT} (Path: {cf.MQTT_PATH})")
    print(f"📌 S3 Endpoint   : {cf.VMS_S3_ENDPOINT} (Bucket: {cf.VMS_S3_BUCKET})")
    print(f"📌 S3 Access Key : {cf.VMS_S3_ACCESS_KEY}")
    print(f"📌 Camera ID/Code: {args.camera_id} / {args.camera_code}")
    print(f"📌 Số người test : {args.people} người")
    print("-" * 70)

    # 1. Kiểm tra cấu hình S3 Secret Key
    if not cf.VMS_S3_SECRET_KEY or "được_cấp_riêng" in cf.VMS_S3_SECRET_KEY:
        log.warning(
            "⚠️  CHÚ Ý: VMS_S3_SECRET_KEY trong file .env chưa được điền Secret Key thật!"
        )
        print("    Vui lòng mở file .env và thay VMS_S3_SECRET_KEY bằng key do quản trị VMS cấp.")
        print("-" * 70)

    # 2. Tạo ảnh snapshot mẫu
    log.info("1️⃣  Đang tạo ảnh JPEG mẫu giả lập phát hiện đám đông...")
    dummy_jpg = create_dummy_snapshot(
        f"TEST CROWD: {args.people} PEOPLE (CAM: {args.camera_code})"
    )
    log.info("   -> Kích thước ảnh: %d bytes (%.1f KB)", len(dummy_jpg), len(dummy_jpg) / 1024)

    # 3. Upload ảnh lên MinIO S3 để lấy object_key
    log.info("2️⃣  Upload ảnh lên MinIO S3 để lấy object_key...")
    upload_start = time.perf_counter()
    object_key = upload_snapshot(dummy_jpg, args.camera_id, category="crowd")
    upload_duration = (time.perf_counter() - upload_start) * 1000

    if not object_key:
        log.error("❌ BƯỚC UPLOAD ẢNH THẤT BẠI! Kiểm tra lại kết nối mạng tới S3 Endpoint và S3 Secret Key.")
        print("=" * 70 + "\n")
        sys.exit(1)

    log.info("   ✅ Upload S3 THÀNH CÔNG! (Thời gian: %.1f ms)", upload_duration)
    log.info("   🔑 Object Key nhận được : %s", object_key)
    log.info(
        "   🔗 URL xem ảnh VMS Backend: http://192.168.1.250:18640/api/v1/snapshot/image?object_key=%s",
        object_key,
    )

    # 4. Đóng gói Event Payload sử dụng object_key vừa nhận được
    log.info("3️⃣  Đang đóng gói Event JSON Payload với snapshot_url = object_key...")
    severity = _crowd_severity(args.people)
    dummy_clusters = [
        {
            "cluster_id": 1,
            "person_count": args.people,
            "detections": list(range(args.people)),
            "track_ids": list(range(1, args.people + 1)),
        }
    ]

    event_payload = {
        "ai_modules": cf.MODULE,
        "submodule_code": cf.MODULE,
        "camera_id": args.camera_id,
        "camera_code": args.camera_code,
        "event_time": datetime.now(TZ).isoformat(timespec="seconds"),
        "event_type": "CROWD_DETECTION",
        "entity_type": "PERSON",
        "severity": severity,
        "payload": {
            "type": "HIGH_DENSITY",
            "zone_id": str(cf.ZONE_ID),
            "person_count": args.people,
            "threshold": int(cf.NUMBER_PERSON_IS_CROWDED),
            "clusters": dummy_clusters,
        },
        "snapshot_url": object_key,
    }

    event_topic = f"smart_vms/ai_events/{cf.MODULE}"
    print("\n📦 CẤU TRÚC JSON PAYLOAD ĐƯỢC TẠO RA:")
    print(json.dumps(event_payload, indent=2, ensure_ascii=False))
    print()

    # 5. Kết nối MQTT Broker
    log.info("4️⃣  Đang kết nối MQTT Broker qua WebSocket (%s:%s)...", cf.MQTT_HOST, cf.MQTT_PORT)
    mqtt_client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"test_pub_{uuid.uuid4().hex[:8]}",
        transport="websockets",
    )
    mqtt_client.ws_set_options(path=cf.MQTT_PATH)
    if cf.MQTT_TLS_ENABLED:
        mqtt_client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    mqtt_client.username_pw_set(cf.MQTT_USER, cf.MQTT_PASS)

    connected_event = [False]

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            connected_event[0] = True
            log.info("   ✅ Kết nối MQTT Broker thành công (rc=0)")
        else:
            log.error("   ❌ Kết nối MQTT Broker thất bại: rc=%s", rc)

    mqtt_client.on_connect = on_connect

    try:
        mqtt_client.connect(cf.MQTT_HOST, cf.MQTT_PORT, keepalive=30)
        mqtt_client.loop_start()

        wait_start = time.time()
        while not connected_event[0] and time.time() - wait_start < 5.0:
            time.sleep(0.1)

        if not connected_event[0]:
            log.error("❌ Hết thời gian chờ kết nối MQTT Broker (5s).")
            sys.exit(1)

        # 6. Publish Crowd Event
        log.info(
            "5️⃣  Publishing Crowd Event (QoS 1) lên topic: %s ...",
            event_topic,
        )
        pub_info = mqtt_client.publish(event_topic, json.dumps(event_payload), qos=1)
        if pub_info.rc != 0:
            log.error("❌ Publish bị reject ngay lập tức (rc=%s)", pub_info.rc)
            sys.exit(1)

        pub_info.wait_for_publish(timeout=5.0)
        if not pub_info.is_published():
            log.error("❌ Hết thời gian chờ Broker ACK (timeout 5s).")
            sys.exit(1)

        log.info("   🎉 BROKER ĐÃ XÁC NHẬN (ACK) SỰ KIỆN THÀNH CÔNG!")
        print("-" * 70)
        print("✨ TỔNG KẾT KIỂM TRA THÀNH CÔNG:")
        print(f"   - Topic              : {event_topic}")
        print(f"   - Mức độ cảnh báo    : {severity} ({args.people} người)")
        print(f"   - Snapshot Object Key: {object_key}")
        print(f"   - Trạng thái VMS     : Đã nhận được sự kiện kèm snapshot URL")

    except Exception:
        log.exception("Lỗi không mong muốn trong quá trình test")
    finally:
        time.sleep(0.5)
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
