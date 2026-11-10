import json
import logging
import threading
from dataclasses import dataclass

from src.config import MODULE

log = logging.getLogger("camera_manager")


def safe_json(raw: bytes) -> dict:
    if not raw or not raw.strip():
        return {}

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        log.warning("payload không phải JSON, bỏ qua")
        return {}


@dataclass(frozen=True)
class Camera:
    id: str
    code: str
    rtsp_url: str


class CameraManager:

    def __init__(self, on_start, on_stop):
        self._lock = threading.Lock()
        self._cameras: dict[str, Camera] = {}
        self._running: dict[str, Camera] = {}
        self.on_start = on_start
        self.on_stop = on_stop

    def get_camera(self, raw: bytes) -> None:
        cams = safe_json(raw=raw).get("cameras") or []
        enabled: dict[str, Camera] = {}
        target_module = MODULE.upper()

        for c in cams:
            if not isinstance(c, dict):
                continue
            raw_modules = c.get("ai_modules") or []
            if isinstance(raw_modules, str):
                raw_modules = [raw_modules]
            module_names = {str(item).strip().upper() for item in raw_modules}
            if target_module not in module_names:
                continue

            code = str(c.get("code") or "").strip()
            cam_id = str(c.get("id") or "").strip()
            restream_urls = c.get("restream_urls")
            rtsp_url = ""
            if isinstance(restream_urls, dict):
                rtsp_url = str(restream_urls.get(target_module) or "").strip()
            if not rtsp_url:
                rtsp_url = str(c.get("rtsp_url") or "").strip()

            if not code or not cam_id or not rtsp_url:
                log.warning(
                    "Bỏ qua camera thiếu thông tin (id/code/rtsp_url): code=%s",
                    code,
                )
                continue

            enabled[code] = Camera(cam_id, code, rtsp_url)

        with self._lock:
            self._cameras = enabled
        log.info("Camera bật module %s: %s", MODULE, list(enabled))
        self._reconcile()

    def _reconcile(self) -> None:
        with self._lock:
            want_codes = set(self._cameras)
            running_codes = set(self._running)

            removed_codes = running_codes - want_codes
            added_codes = want_codes - running_codes
            changed_codes = {
                code
                for code in (want_codes & running_codes)
                if self._cameras[code] != self._running[code]
            }

            stop_codes = sorted(removed_codes | changed_codes)
            start_codes = sorted(added_codes | changed_codes)

            cameras_to_start = {
                code: self._cameras[code]
                for code in start_codes
            }

        for code in stop_codes:
            try:
                self.on_stop(code)
            except Exception:
                log.exception("on_stop failed for camera=%s", code)
            else:
                with self._lock:
                    self._running.pop(code, None)

        for code in start_codes:
            cam = cameras_to_start[code]
            try:
                self.on_start(cam)
            except Exception:
                log.exception("on_start failed for camera=%s", code)
            else:
                with self._lock:
                    self._running[code] = cam

        with self._lock:
            running_snapshot = sorted(self._running)
            cameras_snapshot = sorted(self._cameras)

        log.info(
            "RECONCILE cameras=%s running=%s start=%s stop=%s",
            cameras_snapshot,
            running_snapshot,
            start_codes,
            stop_codes,
        )
