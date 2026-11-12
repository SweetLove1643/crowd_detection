# Bước 6: upload ảnh, publish event + bbox
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import paho.mqtt.client as mqtt
import uuid
import boto3
from botocore.config import Config
import src.config as cf

log = logging.getLogger("publisher")
TZ = timezone(timedelta(hours=7))
_MQTT_PUBLISH_LOCK = threading.Lock()

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
        log.exception("Upload ảnh MinIO S3 lỗi camera=%s", camera_id)
        return None

def _crowd_severity(person_count: int) -> str:
    if person_count >= cf.CROWD_SEVERITY_HIGH_MIN_PERSONS:
        return "HIGH"
    if person_count >= cf.CROWD_SEVERITY_MEDIUM_MIN_PERSONS:
        return "MEDIUM"
    return "LOW"

def publish_crowd_event(
    client: mqtt.Client,
    cam_id: str,
    cam_code: str,
    crowded_clusters: list[dict],
    jpg: bytes | None = None,
) -> bool:
    if not jpg:
        log.error(
            "Crowd event has no snapshot camera=%s; event not published",
            cam_code,
        )
        return False

    object_key = upload_snapshot(jpg, cam_id, category="crowd")
    if not object_key:
        log.error(
            "Crowd event not published because snapshot upload failed camera=%s",
            cam_code,
        )
        return False

    largest_cluster = max(
        (int(c.get("person_count", 0)) for c in crowded_clusters),
        default=0,
    )

    event = {
        "ai_modules": cf.MODULE,
        "submodule_code": cf.MODULE,
        "camera_id": cam_id,
        "camera_code": cam_code,
        "event_time": datetime.now(TZ).isoformat(timespec="seconds"),
        "event_type": "CROWD_DETECTION",
        "entity_type": "PERSON",
        "severity": _crowd_severity(largest_cluster),
        "payload": {
            "type": "HIGH_DENSITY",
            "person_count": sum(
                cluster.get("person_count", 0) for cluster in crowded_clusters
            ),
            "threshold": int(cf.NUMBER_PERSON_IS_CROWDED),
            "clusters": crowded_clusters,
        },
        "snapshot_url": object_key,
    }
    topic = f"smart_vms/ai_events/{cf.MODULE}"

    try:
        info = client.publish(topic, json.dumps(event), qos=1)
        if info.rc != 0:
            log.error(
                "MQTT crowd event rejected camera=%s topic=%s rc=%s",
                cam_code,
                topic,
                info.rc,
            )
            return False

        info.wait_for_publish(timeout=5)
        if not info.is_published():
            log.error(
                "MQTT crowd event timeout camera=%s topic=%s",
                cam_code,
                topic,
            )
            return False

        log.info(
            "MQTT crowd event broker-acknowledged camera=%s topic=%s "
            "clusters=%d object_key=%s",
            cam_code,
            topic,
            len(crowded_clusters),
            object_key,
        )
        return True
    except Exception:
        log.exception(
            "MQTT crowd event exception camera=%s topic=%s",
            cam_code,
            topic,
        )
        return False


def publish_bbox(
    client: mqtt.Client,
    cam_id: str,
    cam_code: str,
    detections: list[dict],
    hud_text: str,
    hud_color: str = "#22c55e",
) -> bool:
    """QoS 0, toạ độ bbox chuẩn hoá 0..1, gửi liên tục theo frame."""
    safe_cam_code = cam_code.translate(
        str.maketrans({"/": "_", "#": "_", "+": "_"})
    )
    topic = f"smart_vms/ai/bbox/{safe_cam_code}"
    msg = {
        "camera_code": cam_code,
        "camera_id": cam_id,
        "organization_id": int(cf.ORG_ID),
        "ai_modules": cf.MODULE,
        "timestamp": time.time(),
        "detections": detections,
        "hud": {
            "text": hud_text,
            "display_ms": 1000,
            "color": hud_color,
        },
    }

    if not client.is_connected():
        return False

    try:
        with _MQTT_PUBLISH_LOCK:
            info = client.publish(topic, json.dumps(msg), qos=0)
        if info.rc != 0:
            log.debug(
                "BBox publish rejected camera=%s topic=%s rc=%s",
                cam_code,
                topic,
                info.rc,
            )
            return False
        return True
    except Exception:
        log.exception(
            "BBox publish exception camera=%s topic=%s",
            cam_code,
            topic,
        )
        return False
