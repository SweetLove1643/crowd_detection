from pathlib import Path

import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO

log = logging.getLogger("person_detector")

class PersonDetector:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence: float = 0.1,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence threshold model error")
        if weights_path is None:
            raise FileNotFoundError(f"Model path failed: {weights_path}")

        self._model = YOLO(str(weights_path))
        self._confidence = confidence
        self._device = 0 if torch.cuda.is_available() else "cpu"
        log.info("Person detector using device=%s", self._device)

        names = self._model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self.person_class_ids = {
            int(class_id)
            for class_id, name in items
            if str(name).strip().casefold() == "person"
        }
        if not self.person_class_ids:
            raise ValueError(f"Model classes not contain 'person': {names}")

    def predict(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Trả về list (x1, y1, x2, y2, conf) theo toạ độ frame gốc."""
        result = self._model.predict(
            frame,
            imgsz=1280,
            conf=self._confidence,
            classes=list(self.person_class_ids),
            device=self._device,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [(float(x1), float(y1), float(x2), float(y2), float(c))
                for (x1, y1, x2, y2), c in zip(xyxy, conf)]

    def track(
        self,
        frame: np.ndarray,
    ) -> list[tuple[float, float, float, float, float, int | None]]:
        """Return person boxes, confidence, and ByteTrack IDs in frame coordinates."""
        result = self._model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=1280,
            conf=self._confidence,
            classes=list(self.person_class_ids),
            device=self._device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [None] * len(boxes)
        )
        return [
            (float(x1), float(y1), float(x2), float(y2), float(score), track_id)
            for (x1, y1, x2, y2), score, track_id in zip(xyxy, conf, track_ids)
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
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            ty = max(p1[1], th + base)  
            cv2.rectangle(out, (p1[0], ty - th - base), (p1[0] + tw, ty), color, -1)
            cv2.putText(out, label, (p1[0], ty - base), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return out
        