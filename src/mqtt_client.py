# Bước 1: kết nối, reconnect, subscribe
import json
import logging
import ssl
import threading

import paho.mqtt.client as mqtt

from src import config
from src.camera_manager import Camera, CameraManager
from src.camera_worker import CameraWorker

log = logging.getLogger("mqtt_client")

TOPIC_CAMERAS = f"smart_vms/cameras/zone/{config.ZONE_ID}"

camera_workers: dict[str, CameraWorker] = {}
workers_lock = threading.Lock()
reconcile_lock = threading.Lock()

mqtt_client: mqtt.Client | None = None


def on_start(
    camera: Camera,
) -> None:
    with workers_lock:
        client = mqtt_client
        if client is None:
            raise RuntimeError("MQTT client chưa được khởi tạo")

        worker = CameraWorker(
            camera=camera,
            client=client,
        )
        camera_workers[camera.code] = worker

        try:
            worker.start()
        except Exception:
            camera_workers.pop(camera.code, None)
            raise

    log.info("STARTED: %s", camera.code)


def on_stop(code: str) -> None:
    with workers_lock:
        worker = camera_workers.pop(code, None)

    if worker is None:
        log.warning("Không tìm thấy worker camera=%s để dừng", code)
        return

    worker.stop()
    log.info("STOP camera=%s", code)


def stop_camera_workers() -> None:
    with workers_lock:
        workers = list(camera_workers.items())
        camera_workers.clear()

    # Báo hiệu dừng đồng loạt cho tất cả camera trước khi join
    for _, worker in workers:
        worker.signal_stop()

    for code, worker in workers:
        worker.stop(timeout=7)
        log.info("STOP camera=%s", code)


camera_manger = CameraManager(on_start, on_stop)


def _handle_camera_payload_async(payload: bytes) -> None:
    def _task() -> None:
        with reconcile_lock:
            try:
                camera_manger.get_camera(payload)
            except Exception:
                log.exception("Reconcile cameras failed")

    threading.Thread(
        target=_task,
        name="camera-reconcile",
        daemon=True,
    ).start()


def build_client() -> mqtt.Client:
    global mqtt_client

    c = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=config.MQTT_CLIENT_ID,
        transport="websockets",
    )
    c.ws_set_options(path=config.MQTT_PATH)
    if config.MQTT_TLS_ENABLED:
        c.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    c.username_pw_set(config.MQTT_USER, config.MQTT_PASS)
    c.reconnect_delay_set(min_delay=1, max_delay=30)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        log.info("Connected: %s", reason_code)
        if reason_code != 0:
            log.error("Failed connect: (%s)", reason_code)
            return

        rc, mid = client.subscribe(TOPIC_CAMERAS, qos=1)
        if rc == 0:
            log.info(
                "Send Subscribe sucessfull!, mid=%s, topics=%s",
                mid,
                TOPIC_CAMERAS,
            )

    def on_disconnect(client, userdata, flags, reason_code, properties=None):
        log.warning("Disconnected: %s (paho sẽ tự reconnect)", reason_code)

    def on_message(client, userdata, msg):
        try:
            log.info(
                "MQTT topic=%s qos=%s retained=%s bytes=%d",
                msg.topic,
                msg.qos,
                msg.retain,
                len(msg.payload),
            )

            if msg.topic == TOPIC_CAMERAS:
                _handle_camera_payload_async(msg.payload)
            else:
                log.warning("TOPIC_CAMERAS not found: %s", msg.topic)

        except Exception:
            log.exception("Processing error: %s", msg.topic)

    def on_subscribe(client, userdata, mid, reason_codes, properties):
        log.info("Subscribe sucessfull!. mid=%s", mid)

    c.on_subscribe = on_subscribe
    c.on_connect = on_connect
    c.on_disconnect = on_disconnect
    c.on_message = on_message

    mqtt_client = c
    return c
