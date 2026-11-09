# Bước 6: upload ảnh, publish event + bbox
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import requests
import paho.mqtt.client as mqtt
import src.config as cf

log = logging.getLogger("publisher")
TZ = timezone(timedelta(hours=7))

def upload_snapshot(jpg: bytes, camera_id: str) -> str | None:
    try:
        r = requests.post(
            cf.STORAGE_URL,
            files={
                "file": ("snap.jpg", jpg, "image/jpeg")
            },
            data={
                "camera_id": camera_id,
                "category": "anomaly" 
            },
            timeout=10,
        )
        r.raise_for_status()
        response = r.json()
        object_key = response.get("object_key") if isinstance(response, dict) else None
        if not object_key or response.get("success") is False:
            log.error(
                "Storage upload unsuccessful camera=%s status=%s has_object_key=%s",
                camera_id,
                r.status_code,
                bool(object_key),
            )
            return None
        log.info(
            "Snapshot uploaded camera=%s object_key=%s",
            camera_id,
            object_key,
        )
        return str(object_key)
    except Exception:
        log.exception("upload ảnh lỗi camera=%s", camera_id)
        return None

def publish_water_event(client: mqtt.Client,
                  cam_id: str, 
                  cam_code: str, 
                  level_cm: float,
                  confidence: float | None = None,
                  zone_name: str | None = None, 
                  jpg: bytes | None = None) -> bool:
    object_key = upload_snapshot(jpg, cam_id) if jpg else None
    event = {
        "ai_modules": cf.MODULE,
        "submodule_code": cf.MODULE,
        "camera_id": cam_id,
        "camera_code": cam_code,
        "event_time": datetime.now(TZ).isoformat(timespec="seconds"),
        "entity_type": "WATER_LEVEL",
        "severity": "MEDIUM",
        "payload": { "type": "HIGH_DENSITY", "person_count": 18, "threshold": 12, "zone_name": "Sảnh" },
    }
    if confidence is not None:
        event["confidence"] = round(confidence, 3)
    if zone_name:
        event["payload"]["zone_name"] = zone_name
    if object_key:
        event["snapshot_url"] = object_key

    topic = f"smart_vms/ai_events/{cf.MODULE}"

    try:
        info = client.publish(topic, json.dumps(event), qos=1)

        if info.rc != 0:
            log.error(
                "MQTT publish failed camera=%s topic=%s rc=%s",
                cam_code,
                topic,
                info.rc,
            )
            return False

        info.wait_for_publish(timeout=5)

        if not info.is_published():
            log.error("MQTT publish timeout camera=%s topic=%s", cam_code, topic)
            return False

        log.info(
            "MQTT event sent camera=%s topic=%s object_key=%s",
            cam_code,
            topic,
            object_key,
        )
        return True

    except Exception:
        log.exception("MQTT publish exception camera=%s topic=%s", cam_code, topic)
        return False

def publish_crowd_event(
    client: mqtt.Client,
    cam_id: str,
    cam_code: str,
    crowded_clusters: list[dict],
    jpg: bytes | None = None,
) -> bool:
    if not jpg:
        log.error("Crowd event has no snapshot camera=%s; event not published", cam_code)
        return False

    object_key = upload_snapshot(jpg, cam_id)
    if not object_key:
        log.error(
            "Crowd event not published because snapshot upload failed camera=%s",
            cam_code,
        )
        return False

    event = {
        "ai_modules": cf.MODULE,
        "submodule_code": cf.MODULE,
        "camera_id": cam_id,
        "camera_code": cam_code,
        "event_time": datetime.now(TZ).isoformat(timespec="seconds"),
        "event_type": "CROWD_DETECTION",
        "entity_type": "PERSON",
        "severity": "HIGH",
        "payload": {
            "type": "HIGH_DENSITY",
            "person_count": sum(cluster.get("person_count", 0) for cluster in crowded_clusters),
            "threshold": int(cf.NUMBER_PERSON_IS_CROWDED),
            "clusters": crowded_clusters,
        },
    }
    event["snapshot_url"] = object_key
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
            log.error("MQTT crowd event timeout camera=%s topic=%s", cam_code, topic)
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
        log.exception("MQTT crowd event exception camera=%s topic=%s", cam_code, topic)
        return False


def publish_bbox(client: mqtt.Client, cam_id: str, cam_code: str,
                 detections: list[dict], hud_text: str,
                 hud_color: str = "#22c55e") -> bool:
    """QoS 0, toạ độ bbox chuẩn hoá 0..1, gửi liên tục theo frame."""
    safe_cam_code = cam_code.translate(str.maketrans({"/": "_", "#": "_", "+": "_"}))
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
    
    try:
        info = client.publish(topic, json.dumps(msg), qos=0)
        if info.rc != 0:
            log.warning(
                "BBox publish rejected camera=%s topic=%s rc=%s",
                cam_code,
                topic,
                info.rc,
            )
            return False
        return True
    except Exception:
        log.exception("BBox publish exception camera=%s topic=%s", cam_code, topic)
        return False
