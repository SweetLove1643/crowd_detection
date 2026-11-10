# src/proximity.py
from __future__ import annotations

from typing import Literal, Sequence

import cv2
import numpy as np

from src import config as cf

Box = Sequence[float]
LengthMode = Literal["max", "height"]


class ProximityEvaluator:

    def __init__(
        self,
        alpha: float | None = None,
        length_mode: LengthMode | None = None,
        y_scale: float | None = None,
    ) -> None:
        resolved_alpha = cf.PROXIMITY_ALPHA if alpha is None else alpha
        resolved_length_mode = (
            cf.PROXIMITY_LENGTH_MODE if length_mode is None else length_mode
        )
        resolved_y_scale = cf.PROXIMITY_Y_SCALE if y_scale is None else y_scale

        if resolved_alpha <= 0:
            raise ValueError("alpha phải > 0")
        if resolved_y_scale <= 0:
            raise ValueError("y_scale phải > 0")
        if resolved_length_mode not in ("max", "height"):
            raise ValueError(
                f"length_mode không hợp lệ: {resolved_length_mode}"
            )

        self.alpha = float(resolved_alpha)
        self.length_mode: LengthMode = resolved_length_mode  # type: ignore[assignment]
        self.y_scale = float(resolved_y_scale)  # >1 để bù phối cảnh nén trục y

    # ---------- helpers ----------
    @staticmethod
    def _to_array(detections: np.ndarray | Sequence[Box]) -> np.ndarray:
        arr = np.asarray(detections, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"Shape không hợp lệ: {arr.shape}")
        return arr[:, :4]

    def _foot_points(self, boxes: np.ndarray) -> np.ndarray:
        x_center = (boxes[:, 0] + boxes[:, 2]) / 2.0
        y_bottom = boxes[:, 3]
        return np.column_stack((x_center, y_bottom))

    def _body_lengths(self, boxes: np.ndarray) -> np.ndarray:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        return np.maximum(w, h) if self.length_mode == "max" else h

    # ---------- public API ----------
    def threshold(self, box_a: Box, box_b: Box) -> float:
        boxes = self._to_array([box_a, box_b])
        la, lb = self._body_lengths(boxes)
        return float(self.alpha * (la + lb) / 2.0)

    def distance(self, box_a: Box, box_b: Box) -> float:
        feet = self._foot_points(self._to_array([box_a, box_b]))
        dx = feet[0, 0] - feet[1, 0]
        dy = (feet[0, 1] - feet[1, 1]) * self.y_scale
        return float(np.hypot(dx, dy))

    def is_neighbor(self, box_a: Box, box_b: Box) -> bool:
        return self.distance(box_a, box_b) <= self.threshold(box_a, box_b)

    def matrix(self, detections: np.ndarray | Sequence[Box]) -> np.ndarray:
        """Ma trận bool (N, N); adj[i, j] = True nếu i, j lân cận. Đường chéo = False."""
        boxes = self._to_array(detections)
        n = len(boxes)
        if n == 0:
            return np.zeros((0, 0), dtype=bool)

        feet = self._foot_points(boxes)
        length = self._body_lengths(boxes)

        diff = feet[:, None, :] - feet[None, :, :]
        diff[..., 1] *= self.y_scale
        dist = np.linalg.norm(diff, axis=-1)                        # (N, N)
        thresh = self.alpha * (length[:, None] + length[None, :]) / 2.0

        adj = dist <= thresh
        np.fill_diagonal(adj, False)
        return adj

    def clusters(self, detections: np.ndarray | Sequence[Box]) -> list[list[int]]:
        """Gom các detection thành connected components theo quan hệ lân cận."""
        adjacency = self.matrix(detections)
        visited = np.zeros(len(adjacency), dtype=bool)
        clusters: list[list[int]] = []

        for start in range(len(adjacency)):
            if visited[start]:
                continue

            visited[start] = True
            stack = [start]
            cluster: list[int] = []
            while stack:
                current = stack.pop()
                cluster.append(current)
                neighbors = np.flatnonzero(adjacency[current] & ~visited)
                visited[neighbors] = True
                stack.extend(neighbors.tolist())

            clusters.append(sorted(cluster))

        return clusters

    def annotate(
        self,
        frame: np.ndarray,
        detections: np.ndarray | Sequence[Box],
        clusters: Sequence[Sequence[int]] | None = None,
        crowd_threshold: int = 3,
    ) -> np.ndarray:
        """Vẽ box đỏ cho thành viên cụm đông, xanh cho các detection còn lại."""
        if crowd_threshold < 1:
            raise ValueError("crowd_threshold phải >= 1")

        boxes = self._to_array(detections)
        annotated = frame.copy()
        if len(boxes) == 0:
            return annotated

        if clusters is None:
            clusters = self.clusters(boxes)

        in_crowd = np.zeros(len(boxes), dtype=bool)
        for cluster in clusters:
            if len(cluster) >= crowd_threshold:
                in_crowd[np.asarray(cluster, dtype=int)] = True

        coordinates = [
            (
                (int(round(float(x1))), int(round(float(y1)))),
                (int(round(float(x2))), int(round(float(y2)))),
            )
            for x1, y1, x2, y2 in boxes
        ]

        for index, (top_left, _) in enumerate(coordinates):
            is_crowd_member = bool(in_crowd[index])
            color = (0, 0, 255) if is_crowd_member else (0, 255, 0)
            label = f"{index}: CROWD" if is_crowd_member else f"{index}: PERSON"
            cv2.putText(
                annotated,
                label,
                (top_left[0], max(top_left[1] - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        for is_crowd_member in (False, True):
            color = (0, 0, 255) if is_crowd_member else (0, 255, 0)
            for index, (top_left, bottom_right) in enumerate(coordinates):
                if bool(in_crowd[index]) == is_crowd_member:
                    cv2.rectangle(annotated, top_left, bottom_right, color, 2)

        return annotated
