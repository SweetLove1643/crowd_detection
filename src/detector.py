import gc
import json
import logging
from pathlib import Path
import shutil
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml
import yaml

from src import config as cf

log = logging.getLogger("person_detector")
_SUPPORTS_QUANTIZE_ARG = "quantize" in DEFAULT_CFG_DICT
_TORCH_RUNTIME_LOCK = threading.Lock()
_TORCH_RUNTIME_CONFIGURED = False
_INFERENCE_SEMAPHORE = threading.Semaphore(cf.MAX_CONCURRENT_INFERENCES)
_SHARED_MODEL_LOCK = threading.Lock()
_MODEL_INFERENCE_LOCK = threading.Lock()
_SHARED_MODELS: dict[tuple[str, str, str, int, bool], tuple[YOLO, dict | list]] = {}
_BYTETRACK_CFG: dict[str, Any] | None = None


def _get_bytetrack_cfg() -> dict[str, Any]:
    global _BYTETRACK_CFG
    if _BYTETRACK_CFG is None:
        with _SHARED_MODEL_LOCK:
            if _BYTETRACK_CFG is None:
                _BYTETRACK_CFG = dict(YAML.load(check_yaml("bytetrack.yaml")))
    return dict(_BYTETRACK_CFG)


def _resolve_openvino_dir(weights_path: Path) -> Path | None:
    configured = Path(cf.OPENVINO_MODEL_PATH)
    if configured.is_dir() and (configured / "metadata.yaml").is_file():
        return configured
    candidate = weights_path.with_name(f"{weights_path.stem}_openvino_model")
    if candidate.is_dir() and (candidate / "metadata.yaml").is_file():
        return candidate
    if weights_path.is_dir() and (weights_path / "metadata.yaml").is_file():
        return weights_path
    return None


def _read_openvino_static_imgsz(ov_dir: Path) -> int | None:
    meta_path = ov_dir / "metadata.yaml"
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            args = meta.get("args") or {}
            is_dynamic = bool(args.get("dynamic", False))
            imgsz = meta.get("imgsz")
            if not is_dynamic and isinstance(imgsz, list) and imgsz:
                return int(imgsz[0])
    except Exception:
        log.debug("Không đọc được metadata.yaml tại %s", meta_path)
    return None


def _resolve_openvino_device(requested_device: str) -> str:
    req = requested_device.strip().lower()
    try:
        import openvino as ov

        available = [d.upper() for d in ov.Core().available_devices]
    except Exception:
        return "intel:cpu"

    if req in {"auto", "intel:auto"}:
        if any(d.startswith("GPU") for d in available):
            return "intel:gpu"
        return "intel:cpu"
    if req.startswith("intel:gpu") and not any(
        d.startswith("GPU") for d in available
    ):
        log.warning(
            "OpenVINO GPU không khả dụng (available=%s), chuyển về intel:cpu",
            available,
        )
        return "intel:cpu"
    if not req.startswith("intel:"):
        return f"intel:{req}"
    return req


def _read_engine_metadata(engine_path: Path) -> dict[str, Any]:
    """Đọc metadata JSON nhúng ở đầu file TensorRT .engine do Ultralytics export."""
    try:
        file_size = engine_path.stat().st_size
        if file_size < 4:
            return {}
        with engine_path.open("rb") as f:
            header_len = int.from_bytes(f.read(4), byteorder="little")
            if 0 < header_len <= file_size - 4:
                meta = json.loads(f.read(header_len).decode("utf-8"))
                if isinstance(meta, dict):
                    return meta
    except Exception:
        log.debug("Không đọc được metadata header từ %s", engine_path)
    return {}


def _read_engine_static_imgsz(engine_path: Path) -> int | None:
    meta = _read_engine_metadata(engine_path)
    if not meta:
        return None
    args = meta.get("args") or {}
    is_dynamic = bool(args.get("dynamic", False)) if isinstance(args, dict) else False
    if is_dynamic:
        return None
    imgsz = meta.get("imgsz")
    if isinstance(imgsz, list) and imgsz:
        val = int(imgsz[0])
        return val if val > 0 else None
    if isinstance(imgsz, (int, float)) and int(imgsz) > 0:
        return int(imgsz)
    return None


def _resolve_engine_imgsz(engine_path: Path, requested_imgsz: int) -> int:
    static_imgsz = _read_engine_static_imgsz(engine_path)
    if static_imgsz is not None and static_imgsz != requested_imgsz:
        log.warning(
            "File TensorRT %s được export với imgsz=%d khác YOLO_IMGSZ=%d; "
            "đang dùng imgsz=%d (hãy xóa file .engine và khởi động lại nếu muốn tự export lại theo YOLO_IMGSZ=%d)",
            engine_path,
            static_imgsz,
            requested_imgsz,
            static_imgsz,
            requested_imgsz,
        )
        return static_imgsz
    return requested_imgsz


def _build_tensorrt_export_kwargs(imgsz: int, half: bool) -> dict[str, Any]:
    export_kwargs: dict[str, Any] = {
        "format": "engine",
        "imgsz": int(imgsz),
        "device": 0,
        "batch": 1,
        "simplify": True,
    }
    if half:
        if _SUPPORTS_QUANTIZE_ARG:
            export_kwargs["quantize"] = 16
        else:
            export_kwargs["half"] = True
    elif not _SUPPORTS_QUANTIZE_ARG:
        export_kwargs["half"] = False
    return export_kwargs


def _export_tensorrt_engine(
    pt_path: Path,
    engine_path: Path,
    imgsz: int,
    half: bool,
) -> Path:
    """Tự động convert model PyTorch (.pt) sang TensorRT (.engine) trên GPU hiện tại."""
    log.info(
        "Chưa có file TensorRT tại %s -> bắt đầu tự động convert từ %s "
        "(imgsz=%d, fp16=%s, device=0, lần đầu có thể mất 1-5 phút)...",
        engine_path,
        pt_path,
        imgsz,
        half,
    )
    started = time.perf_counter()
    pt_model: YOLO | None = None
    try:
        pt_model = YOLO(str(pt_path))
        exported = Path(
            pt_model.export(**_build_tensorrt_export_kwargs(imgsz, half))
        )
        if exported.is_file() and exported.resolve() != engine_path.resolve():
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported), str(engine_path))
        if not engine_path.is_file() or engine_path.stat().st_size <= 0:
            raise RuntimeError(
                f"TensorRT export finished but output engine missing or empty: {engine_path}"
            )
        elapsed_s = time.perf_counter() - started
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        log.info(
            "Đã export TensorRT thành công: %s (%.1f MB, %.1fs)",
            engine_path,
            size_mb,
            elapsed_s,
        )
        return engine_path
    except Exception:
        if engine_path.is_file():
            try:
                engine_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        del pt_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PersonDetector:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence: float | None = None,
    ) -> None:
        resolved_weights = (
            cf.WEIGHT_YOLO_MODEL_PATH if weights_path is None else weights_path
        )
        resolved_conf = (
            float(cf.YOLO_MODEL_CONF)
            if confidence is None
            else float(confidence)
        )
        if not 0.0 <= resolved_conf <= 1.0:
            raise ValueError("Confidence threshold model error")
        if resolved_weights is None:
            raise FileNotFoundError(f"Model path failed: {resolved_weights}")

        weights_path = Path(resolved_weights)
        self._confidence = resolved_conf
        self._tracker = BYTETracker(
            IterableSimpleNamespace(**_get_bytetrack_cfg())
        )
        self._configure_torch_runtime()

        with _SHARED_MODEL_LOCK:
            self._backend, self._device, model_target, self._imgsz, self._half = (
                self._select_runtime(weights_path)
            )
            model_key = (
                self._backend,
                str(self._device),
                str(model_target.resolve()),
                self._imgsz,
                self._half,
            )
            cached = _SHARED_MODELS.get(model_key)
            if cached is None:
                if self._backend == "openvino":
                    model = YOLO(str(model_target), task="detect")
                    names = self._load_openvino_names(model, model_target)
                elif self._backend == "tensorrt":
                    model = YOLO(str(model_target), task="detect")
                    names = self._load_engine_names(model, model_target)
                else:
                    model = YOLO(str(model_target))
                    names = model.names
                _SHARED_MODELS[model_key] = (model, names)
                log.info(
                    "Loaded shared YOLO model backend=%s device=%s model=%s "
                    "imgsz=%d half=%s cpu_threads=%d max_concurrent=%d",
                    self._backend,
                    self._device,
                    model_target,
                    self._imgsz,
                    self._half,
                    torch.get_num_threads(),
                    cf.MAX_CONCURRENT_INFERENCES,
                )
            else:
                model, names = cached
                log.info(
                    "Reusing shared YOLO model backend=%s device=%s model=%s",
                    self._backend,
                    self._device,
                    model_target,
                )

        self._model = model
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self.person_class_ids = {
            int(class_id)
            for class_id, name in items
            if str(name).strip().casefold() == "person"
        }
        if not self.person_class_ids:
            raise ValueError(f"Model classes not contain 'person': {names}")

    def _select_runtime(
        self,
        weights_path: Path,
    ) -> tuple[str, int | str, Path, int, bool]:
        requested_backend = cf.YOLO_INFERENCE_BACKEND.strip().lower()
        has_cuda = torch.cuda.is_available()

        # 1. Ưu tiên GPU NVIDIA (ví dụ RTX A400 4GB VRAM trên môi trường MVP)
        if has_cuda and requested_backend in {"auto", "pytorch", "cuda", "tensorrt"}:
            is_engine_target = weights_path.suffix.lower() == ".engine"
            engine_path = (
                weights_path
                if is_engine_target
                else weights_path.with_suffix(".engine")
            )
            pt_path = weights_path.with_suffix(".pt")

            if requested_backend == "tensorrt":
                if engine_path.is_file():
                    return (
                        "tensorrt",
                        0,
                        engine_path,
                        _resolve_engine_imgsz(engine_path, cf.YOLO_IMGSZ),
                        cf.YOLO_HALF_PRECISION,
                    )
                if not pt_path.is_file():
                    raise FileNotFoundError(
                        f"TensorRT engine not found: {engine_path}, and source PyTorch "
                        f"weights not found for auto-export: {pt_path}"
                    )
                try:
                    exported_engine = _export_tensorrt_engine(
                        pt_path,
                        engine_path,
                        cf.YOLO_IMGSZ,
                        cf.YOLO_HALF_PRECISION,
                    )
                    return (
                        "tensorrt",
                        0,
                        exported_engine,
                        _resolve_engine_imgsz(exported_engine, cf.YOLO_IMGSZ),
                        cf.YOLO_HALF_PRECISION,
                    )
                except Exception:
                    log.exception(
                        "Auto-export TensorRT thất bại từ %s sang %s",
                        pt_path,
                        engine_path,
                    )
                    log.warning(
                        "Fallback sang PyTorch CUDA với model nguồn %s",
                        pt_path,
                    )
                    return (
                        "pytorch",
                        0,
                        pt_path,
                        cf.YOLO_IMGSZ,
                        cf.YOLO_HALF_PRECISION,
                    )

            if is_engine_target and engine_path.is_file():
                return (
                    "tensorrt",
                    0,
                    engine_path,
                    _resolve_engine_imgsz(engine_path, cf.YOLO_IMGSZ),
                    cf.YOLO_HALF_PRECISION,
                )
            if not weights_path.exists():
                raise FileNotFoundError(f"Model path not found: {weights_path}")
            return (
                "pytorch",
                0,
                weights_path,
                cf.YOLO_IMGSZ,
                cf.YOLO_HALF_PRECISION,
            )

        # 2. Nếu bật tensorrt trên máy không có CUDA -> fallback về PyTorch CPU
        if not has_cuda and requested_backend == "tensorrt":
            fallback_path = (
                weights_path.with_suffix(".pt")
                if weights_path.suffix.lower() == ".engine"
                else weights_path
            )
            log.warning(
                "CUDA không khả dụng cho YOLO_INFERENCE_BACKEND='tensorrt', "
                "fallback về PyTorch CPU với %s",
                fallback_path,
            )
            if not fallback_path.exists():
                raise FileNotFoundError(
                    f"CUDA unavailable for TensorRT and PyTorch fallback weights not found: {fallback_path}"
                )
            return ("pytorch", "cpu", fallback_path, cf.YOLO_IMGSZ, False)

        # 3. Trên máy CPU-only (ví dụ i7-8650U): ưu tiên OpenVINO nếu backend=auto/openvino
        if requested_backend in {"auto", "openvino"}:
            ov_dir = _resolve_openvino_dir(weights_path)
            if ov_dir is not None:
                ov_device = _resolve_openvino_device(cf.YOLO_OPENVINO_DEVICE)
                static_imgsz = _read_openvino_static_imgsz(ov_dir)
                imgsz = static_imgsz or cf.YOLO_IMGSZ
                return ("openvino", ov_device, ov_dir, imgsz, False)
            if requested_backend == "openvino":
                log.warning(
                    "Không tìm thấy thư mục model OpenVINO cho %s, fallback về PyTorch",
                    weights_path,
                )

        # 4. Fallback về PyTorch (CPU hoặc CUDA)
        fallback_path = weights_path
        if not has_cuda and weights_path.suffix.lower() == ".engine":
            fallback_path = weights_path.with_suffix(".pt")
            log.warning(
                "CUDA không khả dụng để chạy %s, fallback sang PyTorch CPU với %s",
                weights_path,
                fallback_path,
            )
        if not fallback_path.exists():
            raise FileNotFoundError(f"Model path not found: {fallback_path}")
        device: int | str = 0 if has_cuda else "cpu"
        half = bool(has_cuda and cf.YOLO_HALF_PRECISION)
        return ("pytorch", device, fallback_path, cf.YOLO_IMGSZ, half)

    @staticmethod
    def _load_engine_names(model: YOLO, engine_path: Path) -> dict | list:
        meta = _read_engine_metadata(engine_path)
        if meta.get("names"):
            return meta["names"]
        return model.names

    @staticmethod
    def _load_openvino_names(model: YOLO, ov_dir: Path) -> dict | list:
        meta_path = ov_dir / "metadata.yaml"
        if meta_path.is_file():
            try:
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and meta.get("names"):
                    return meta["names"]
            except Exception:
                pass
        return model.names

    def _configure_torch_runtime(self) -> None:
        global _TORCH_RUNTIME_CONFIGURED
        if _TORCH_RUNTIME_CONFIGURED:
            return

        with _TORCH_RUNTIME_LOCK:
            if _TORCH_RUNTIME_CONFIGURED:
                return
            torch.set_num_threads(cf.YOLO_CPU_THREADS)
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
            _TORCH_RUNTIME_CONFIGURED = True

    def _restore_torch_runtime(self) -> None:
        if torch.get_num_threads() == cf.YOLO_CPU_THREADS:
            return

        with _TORCH_RUNTIME_LOCK:
            if torch.get_num_threads() != cf.YOLO_CPU_THREADS:
                torch.set_num_threads(cf.YOLO_CPU_THREADS)

    def _precision_kwargs(self) -> dict[str, Any]:
        if not self._half:
            return {}
        if _SUPPORTS_QUANTIZE_ARG:
            return {"quantize": 16}
        return {"half": True}

    def predict(
        self,
        frame: np.ndarray,
    ) -> list[tuple[float, float, float, float, float]]:
        """Trả về list (x1, y1, x2, y2, conf) theo toạ độ frame gốc."""
        infer_kwargs = self._precision_kwargs()
        with _INFERENCE_SEMAPHORE, _MODEL_INFERENCE_LOCK:
            result = self._model.predict(
                frame,
                imgsz=self._imgsz,
                conf=self._confidence,
                iou=cf.YOLO_MODEL_IOU,
                classes=list(self.person_class_ids),
                device=self._device,
                verbose=False,
                **infer_kwargs,
            )[0]
            self._restore_torch_runtime()

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [
            (float(x1), float(y1), float(x2), float(y2), float(c))
            for (x1, y1, x2, y2), c in zip(xyxy, conf)
        ]

    def track(
        self,
        frame: np.ndarray,
        latest_frame_getter: Any = None,
    ) -> list[tuple[float, float, float, float, float, int | None]]:
        """Return person boxes, confidence, and ByteTrack IDs in frame coordinates."""
        infer_kwargs = self._precision_kwargs()
        with _INFERENCE_SEMAPHORE, _MODEL_INFERENCE_LOCK:
            if latest_frame_getter is not None:
                newer_frame = latest_frame_getter()
                if newer_frame is not None:
                    frame = newer_frame

            result = self._model.predict(
                frame,
                imgsz=self._imgsz,
                conf=self._confidence,
                iou=cf.YOLO_MODEL_IOU,
                classes=list(self.person_class_ids),
                device=self._device,
                verbose=False,
                **infer_kwargs,
            )[0]

            self._restore_torch_runtime()

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            if boxes is not None:
                self._tracker.update(boxes.cpu().numpy(), frame)
            return []

        det_boxes = boxes.cpu().numpy()
        tracks = self._tracker.update(det_boxes, frame)
        if len(tracks) == 0:
            if any(not t.is_activated for t in self._tracker.tracked_stracks):
                return []
            xyxy = det_boxes.xyxy
            conf = det_boxes.conf
            return [
                (float(x1), float(y1), float(x2), float(y2), float(score), None)
                for (x1, y1, x2, y2), score in zip(xyxy, conf)
            ]

        return [
            (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[5]),
                int(row[4]),
            )
            for row in tracks
        ]

    def visualize(
        self,
        frame: np.ndarray,
        detections: list[tuple[float, float, float, float, float]],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Vẽ bbox + conf lên bản copy của frame, trả về frame đã vẽ."""
        out = frame.copy()
        for x1, y1, x2, y2, conf in detections:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(out, p1, p2, color, thickness)

            label = f"people {conf:.2f}"
            (tw, th), base = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            ty = max(p1[1], th + base)
            cv2.rectangle(
                out, (p1[0], ty - th - base), (p1[0] + tw, ty), color, -1
            )
            cv2.putText(
                out,
                label,
                (p1[0], ty - base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return out
