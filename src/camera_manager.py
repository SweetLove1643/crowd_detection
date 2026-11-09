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
@dataclass
class Camera:
    id: str
    code: str
    rtsp_url: str
    
class CameraManager:

    def __init__(self, on_start, on_stop):
        self._lock = threading.Lock()
        self._cameras: dict[str, Camera] = {}
        self._running: set[str] = set()
        self.on_start = on_start
        self.on_stop = on_stop

    def get_camera(self, raw: bytes) -> None:
        cams = safe_json(raw=raw).get("cameras") or []
        enabled = {}
        for c in cams:
            raw_modules = c.get("ai_modules") or []
            if isinstance(raw_modules, str):
                raw_modules = [raw_modules]
            module_names = {str(item).strip().upper() for item in raw_modules}
            if MODULE.upper() not in module_names:
                continue
            code = str(c["code"]).strip()
            if not code:
                continue
            enabled[code] = Camera(str(c["id"]), code, str(c["rtsp_url"]))
        with self._lock:
            self._cameras =enabled
        log.info("Camera bật module %s: %s", MODULE, list(enabled))
        self._reconcile()

    def _reconcile(self) -> None:
        with self._lock:
            want = set(self._cameras)
            running = set(self._running)

            start = want - self._running
            stop = self._running - want   

            cameras_to_start = {
                code: self._cameras[code]
                for code in start
            }

        for code in stop:
            try:
                self.on_stop(code)
            except Exception:
                log.exception("on_stop failed for camera=%s", code)
            else:
                with self._lock:
                    self._running.discard(code)
        for code in start:
            try:
                self.on_start(cameras_to_start[code])
            except Exception:
                log.exception("on_start failed for camera=%s", code)
            else:
                with self._lock:
                    self._running.add(code)

        with self._lock:
            running_snapshot = sorted(self._running)
            cameras_snapshot = sorted(self._cameras)

        log.info(
            "RECONCILE cameras=%s running=%s start=%s stop=%s",
            cameras_snapshot,
            running_snapshot,
            sorted(start),
            sorted(stop),
        )



