# Bước 3: 1 camera = 1 thread, đọc RTSP
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from src.detector import PersonDetector
from src.analyze import ProximityEvaluator
import src.config as cf
from src.camera_manager import Camera
from src.publisher import publish_bbox, publish_crowd_event

log = logging.getLogger("camera_worker")

OPEN_TIMEOUT_MS = 5_000
READ_TIMEOUT_MS = 5_000
RECONNECT_DELAY_S = 3
DEBUG_FRAME_SAVE_INTERVAL_S = 5.0
PERFORMANCE_LOG_INTERVAL_S = 5.0
DEBUG_VIDEO_FPS = 10.0
CROWD_TRACK_GRACE_S = 1.0

Detection = tuple[float, float, float, float, float, int | None]
Cluster = list[int]
CrowdedCluster = tuple[int, Cluster]


@dataclass
class _CrowdCandidate:
    track_ids: set[int]
    started_at: float
    last_seen: float
    notified: bool = False


class CrowdPersistence:
    def __init__(
        self,
        duration_s: float,
        grace_s: float = CROWD_TRACK_GRACE_S,
    ) -> None:
        self.duration_s = duration_s
        self.grace_s = grace_s
        self._candidates: list[_CrowdCandidate] = []

    def update(
        self,
        crowded_clusters: list[CrowdedCluster],
        detections: list[Detection],
        now: float,
    ) -> list[tuple[int, Cluster, set[int]]]:
        self._candidates = [
            candidate
            for candidate in self._candidates
            if now - candidate.last_seen <= self.grace_s
        ]
        available = set(range(len(self._candidates)))
        ready: list[tuple[int, list[int], set[int]]] = []

        for cluster_index, cluster in crowded_clusters:
            tracked_ids = {
                int(detections[index][5])
                for index in cluster
                if detections[index][5] is not None
            }
            if not tracked_ids:
                continue

            match_index = None
            best_overlap = -1
            for index in sorted(available):
                overlap = len(self._candidates[index].track_ids & tracked_ids)
                if overlap > best_overlap:
                    best_overlap = overlap
                    match_index = index

            if match_index is not None:
                candidate = self._candidates[match_index]
                if candidate.track_ids & tracked_ids:
                    available.discard(match_index)
                    candidate.track_ids = tracked_ids
                    candidate.last_seen = now
                else:
                    match_index = None

            if match_index is None:
                candidate = _CrowdCandidate(set(tracked_ids), now, now)
                self._candidates.append(candidate)
            
            if not candidate.notified and now - candidate.started_at >= self.duration_s:
                ready.append((cluster_index, cluster, set(tracked_ids)))

        return ready

    def mark_notified(self, track_ids: set[int]) -> None:
        for candidate in self._candidates:
            if candidate.track_ids & track_ids:
                candidate.notified = True


def _build_bbox_detections(
    detections: list[Detection],
    clusters: list[Cluster],
    crowded_clusters: list[CrowdedCluster],
    frame_width: int,
    frame_height: int,
    publish_all_person_bbox: bool,
    padding_percent: float,
) -> list[dict[str, object]]:
    is_crowded = bool(crowded_clusters)
    detection_clusters = {
        detection_index: cluster_index + 1
        for cluster_index, cluster in enumerate(clusters)
        for detection_index in cluster
    }
    crowd_detection_indices = {
        detection_index
        for _, cluster in crowded_clusters
        for detection_index in cluster
    }
    bbox_detections: list[dict[str, object]] = []

    for detection_index, detection in enumerate(detections):
        is_crowd_member = detection_index in crowd_detection_indices
        if not is_crowd_member and (is_crowded or not publish_all_person_bbox):
            continue

        x1, y1, x2, y2, confidence, track_id = detection
        person_bbox: dict[str, object] = {
            "id": (
                f"person_{track_id}"
                if track_id is not None
                else f"person_{detection_index}"
            ),
            "class": "person",
            "confidence": round(confidence, 4),
            "bbox": [
                round(x1 / frame_width, 6),
                round(y1 / frame_height, 6),
                round(x2 / frame_width, 6),
                round(y2 / frame_height, 6),
            ],
            "color": "#ef4444" if is_crowd_member else "#22c55e",
            "cluster_id": detection_clusters[detection_index],
        }
        if track_id is not None:
            person_bbox["track_id"] = track_id
        bbox_detections.append(person_bbox)

    for cluster_index, cluster in crowded_clusters:
        cluster_boxes = [detections[index] for index in cluster]
        padding = padding_percent / 100
        x1 = min(box[0] for box in cluster_boxes)
        y1 = min(box[1] for box in cluster_boxes)
        x2 = max(box[2] for box in cluster_boxes)
        y2 = max(box[3] for box in cluster_boxes)
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_width, x2 + pad_x)
        y2 = min(frame_height, y2 + pad_y)
        bbox_detections.append(
            {
                "id": f"crowd_{cluster_index + 1}",
                "class": "crowd",
                "bbox": [
                    round(x1 / frame_width, 6),
                    round(y1 / frame_height, 6),
                    round(x2 / frame_width, 6),
                    round(y2 / frame_height, 6),
                ],
                "color": "#ef4444",
                "cluster_id": cluster_index + 1,
                "person_count": len(cluster),
                "track_ids": sorted(
                    detections[index][5]
                    for index in cluster
                    if detections[index][5] is not None
                ),
            }
        )

    return bbox_detections


def _annotate_bbox_detections(
    frame: np.ndarray,
    detections: list[Detection],
    crowded_clusters: list[CrowdedCluster],
    padding_percent: float,
    inference_ms: float,
    processing_fps: float,
) -> np.ndarray:
    annotated = frame.copy()
    crowd_indices = {
        detection_index
        for _, cluster in crowded_clusters
        for detection_index in cluster
    }

    def draw_box(
        box: tuple[float, float, float, float],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    for detection_index, (
        x1,
        y1,
        x2,
        y2,
        confidence,
        track_id,
    ) in enumerate(detections):
        is_crowd_member = detection_index in crowd_indices
        draw_box(
            (x1, y1, x2, y2),
            (0, 0, 255) if is_crowd_member else (0, 255, 0),
            (
                f"person_{track_id}"
                if track_id is not None
                else f"person_{detection_index}"
            )
            + f" {confidence:.2f}",
        )

    for cluster_index, cluster in crowded_clusters:
        cluster_boxes = [detections[index] for index in cluster]
        x1 = min(box[0] for box in cluster_boxes)
        y1 = min(box[1] for box in cluster_boxes)
        x2 = max(box[2] for box in cluster_boxes)
        y2 = max(box[3] for box in cluster_boxes)
        pad_x = (x2 - x1) * padding_percent / 100
        pad_y = (y2 - y1) * padding_percent / 100
        draw_box(
            (
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(frame.shape[1], x2 + pad_x),
                min(frame.shape[0], y2 + pad_y),
            ),
            (0, 0, 255),
            f"crowd_{cluster_index + 1} ({len(cluster)} people)",
        )

    cv2.putText(
        annotated,
        f"Inference: {inference_ms:.1f} ms | Processing: {processing_fps:.1f} FPS",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _publish_crowd_event_async(
    client: mqtt.Client,
    camera_id: str,
    camera_code: str,
    clusters: list[dict],
    snapshot_jpg: bytes | None,
) -> bool:
    thread = threading.Thread(
        target=publish_crowd_event,
        args=(client, camera_id, camera_code, clusters),
        kwargs={"jpg": snapshot_jpg},
        name=f"crowd-event-{camera_code}",
        daemon=True,
    )
    try:
        thread.start()
    except RuntimeError:
        log.exception("Could not start crowd event publisher camera=%s", camera_code)
        return False
    return True


class CameraWorker:
    def __init__(
        self,
        camera: Camera,
        client: mqtt.Client,
    ) -> None:
        self.camera = camera
        self.client = client
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"Woker for camera {self.camera.code} started")

        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.camera.code}",
            daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 7) -> None:
        self._stop_event.set()

        if self._thread is None:
            return

        self._thread.join(timeout=timeout)

        if timeout is not None and self._thread.is_alive():
            log.warning("Worker camera=%s haven't stopped yet %.1f second", self.camera.code, timeout)

    def _run(self) -> None:

        weight_path = Path(cf.WEIGHT_YOLO_MODEL_PATH)
        confidence = float(cf.YOLO_MODEL_CONF)

        detector = PersonDetector(weights_path=weight_path, confidence=confidence)
        analyzer = ProximityEvaluator()
        crowd_persistence = CrowdPersistence(cf.CROWD_PERSISTENCE_SECONDS)
        was_crowded = False
        last_cluster_log_time = 0.0
        last_snapshot_time = 0.0
        last_performance_log_time = time.monotonic()
        last_frame_time: float | None = None
        frame_count = 0
        inference_time_total_ms = 0.0

        image_save_dir = Path(__file__).resolve().parents[1] / "resources" / "img_test"
        image_save_dir.mkdir(parents=True, exist_ok=True)
        debug_video_dir = Path(__file__).resolve().parents[1] / "resources" / "debug_videos"
        debug_video_dir.mkdir(parents=True, exist_ok=True)
        debug_writer = None
        debug_video_started_at: float | None = None
        debug_video_frame_count = 0
        debug_video_disabled = False
        debug_video_path: Path | None = None

        while not self._stop_event.is_set():
            capture = None

            try:
                capture = cv2.VideoCapture(
                    "./resources/recordings/nhatron-loira_20261105_163141.mp4",
                    # self.camera.rtsp_url,
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        OPEN_TIMEOUT_MS,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        READ_TIMEOUT_MS,
                    ],
                )

                if not capture.isOpened():
                    log.warning("Cannot open RTSP camera=%s", self.camera.code)
                else:
                    log.info("Opened RTSP camera=%s", self.camera.code)

                    while not self._stop_event.is_set():
                        ok, frame = capture.read()

                        if not ok or frame is None:
                            log.warning(
                                "Cannot read frame: camera=%s, ok=%s, frame_is_none=%s",
                                self.camera.code,
                                ok,
                                frame is None,
                            )
                            break

                        if debug_writer is None and not debug_video_disabled:
                            safe_camera_code = "".join(
                                char if char.isalnum() or char in "-_" else "_"
                                for char in self.camera.code
                            )
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            debug_video_path = (
                                debug_video_dir
                                / f"{safe_camera_code}_{timestamp}.mp4"
                            )
                            frame_height, frame_width = frame.shape[:2]
                            debug_writer = cv2.VideoWriter(
                                str(debug_video_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                DEBUG_VIDEO_FPS,
                                (frame_width, frame_height),
                            )
                            if debug_writer.isOpened():
                                debug_video_started_at = time.monotonic()
                                log.info(
                                    "Debug video started camera=%s path=%s fps=%.1f",
                                    self.camera.code,
                                    debug_video_path,
                                    DEBUG_VIDEO_FPS,
                                )
                            else:
                                log.error(
                                    "Cannot create debug video camera=%s path=%s",
                                    self.camera.code,
                                    debug_video_path,
                                )
                                debug_writer.release()
                                debug_writer = None
                                debug_video_disabled = True

                        frame_started_at = time.monotonic()
                        processing_fps = (
                            1 / (frame_started_at - last_frame_time)
                            if last_frame_time is not None
                            and frame_started_at > last_frame_time
                            else 0.0
                        )
                        last_frame_time = frame_started_at
                        inference_started = time.perf_counter()
                        dets = detector.track(frame)
                        inference_ms = (time.perf_counter() - inference_started) * 1000
                        clusters = analyzer.clusters(dets)
                        crowded_clusters = [
                            (cluster_index, cluster)
                            for cluster_index, cluster in enumerate(clusters)
                            if len(cluster) >= cf.NUMBER_PERSON_IS_CROWDED
                        ]
                        is_crowded = bool(crowded_clusters)

                        now = time.monotonic()
                        ready_clusters = crowd_persistence.update(
                            crowded_clusters,
                            dets,
                            now,
                        )
                        if now - last_cluster_log_time >= 1.0:
                            cluster_details = [
                                {
                                    "cluster": cluster_index + 1,
                                    "people": len(cluster),
                                    "detections": cluster,
                                    "crowd": len(cluster) >= cf.NUMBER_PERSON_IS_CROWDED,
                                    "track_ids": [
                                        dets[index][5]
                                        for index in cluster
                                        if dets[index][5] is not None
                                    ],
                                }
                                for cluster_index, cluster in enumerate(clusters)
                            ]
                            log.info(
                                "Cluster summary camera=%s people=%d clusters=%d "
                                "crowd_threshold>=%d details=%s",
                                self.camera.code,
                                len(dets),
                                len(clusters),
                                cf.NUMBER_PERSON_IS_CROWDED,
                                cluster_details,
                            )
                            last_cluster_log_time = now

                        if is_crowded and not was_crowded:
                            log.info(
                                "Crowd candidate detected camera=%s; waiting %.1fs persistence",
                                self.camera.code,
                                cf.CROWD_PERSISTENCE_SECONDS,
                            )
                        elif was_crowded and not is_crowded:
                            log.info("Crowd cleared camera=%s", self.camera.code)
                        was_crowded = is_crowded

                        frame_height, frame_width = frame.shape[:2]
                        bbox_detections = _build_bbox_detections(
                            dets,
                            clusters,
                            crowded_clusters,
                            frame_width,
                            frame_height,
                            cf.PUBLISH_ALL_PERSON_BBOX,
                            cf.CROWD_BBOX_PADDING_PERCENT,
                        )
                        debug_frame = _annotate_bbox_detections(
                            frame,
                            dets,
                            crowded_clusters,
                            cf.CROWD_BBOX_PADDING_PERCENT,
                            inference_ms,
                            processing_fps,
                        )

                        if ready_clusters:
                            log.warning(
                                "Persistent crowd detected camera=%s duration=%.1fs clusters=%s",
                                self.camera.code,
                                cf.CROWD_PERSISTENCE_SECONDS,
                                [cluster for _, cluster, _ in ready_clusters],
                            )
                            snapshot_jpg = None
                            try:
                                encoded, snapshot_buffer = cv2.imencode(
                                    ".jpg",
                                    debug_frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                                )
                                if encoded:
                                    snapshot_jpg = snapshot_buffer.tobytes()
                                else:
                                    log.error(
                                        "Failed to encode crowd snapshot camera=%s",
                                        self.camera.code,
                                    )
                            except cv2.error:
                                log.exception(
                                    "Failed to encode crowd snapshot camera=%s",
                                    self.camera.code,
                                )
                            event_scheduled = _publish_crowd_event_async(
                                self.client,
                                self.camera.id,
                                self.camera.code,
                                [
                                    {
                                        "cluster_id": cluster_index + 1,
                                        "person_count": len(cluster),
                                        "detections": cluster,
                                        "track_ids": sorted(track_ids),
                                    }
                                    for cluster_index, cluster, track_ids in ready_clusters
                                ],
                                snapshot_jpg=snapshot_jpg,
                            )
                            if not event_scheduled:
                                log.error(
                                    "Crowd event was not scheduled camera=%s; "
                                    "will retry while crowd persists",
                                    self.camera.code,
                                )
                            else:
                                for _, _, track_ids in ready_clusters:
                                    crowd_persistence.mark_notified(track_ids)

                        hud_text = (
                            f"CROWD: {len(crowded_clusters)} cluster(s)"
                            if is_crowded
                            else f"People: {len(dets)}"
                        )
                        bbox_published = publish_bbox(
                            self.client,
                            self.camera.id,
                            self.camera.code,
                            bbox_detections,
                            hud_text,
                            "#ef4444" if is_crowded else "#22c55e",
                        )

                        frame_count += 1
                        inference_time_total_ms += inference_ms
                        if now - last_performance_log_time >= PERFORMANCE_LOG_INTERVAL_S:
                            elapsed = now - last_performance_log_time
                            log.info(
                                "Realtime performance camera=%s frames=%d "
                                "fps=%.2f avg_inference_ms=%.1f "
                                "bbox_count=%d bbox_publish_ok=%s",
                                self.camera.code,
                                frame_count,
                                frame_count / elapsed,
                                inference_time_total_ms / frame_count,
                                len(bbox_detections),
                                bbox_published,
                            )
                            frame_count = 0
                            inference_time_total_ms = 0.0
                            last_performance_log_time = now

                        if now - last_snapshot_time >= DEBUG_FRAME_SAVE_INTERVAL_S:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            safe_camera_code = "".join(
                                char if char.isalnum() or char in "-_" else "_"
                                for char in self.camera.code
                            )
                            img_path = image_save_dir / f"{safe_camera_code}_{ts}.jpg"
                            ok = cv2.imwrite(
                                str(img_path),
                                debug_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90],
                            )
                            if not ok:
                                log.error(
                                    "Failed to save debug snapshot camera=%s path=%s",
                                    self.camera.code,
                                    img_path,
                                )
                            else:
                                log.debug(
                                    "Saved debug snapshot camera=%s path=%s bboxes=%d",
                                    self.camera.code,
                                    img_path,
                                    len(dets) + len(crowded_clusters),
                                )
                            last_snapshot_time = now

                        if debug_writer is not None and debug_video_started_at is not None:
                            elapsed = time.monotonic() - debug_video_started_at
                            target_frame_count = max(
                                debug_video_frame_count + 1,
                                int(round(elapsed * DEBUG_VIDEO_FPS)),
                            )
                            for _ in range(target_frame_count - debug_video_frame_count):
                                debug_writer.write(debug_frame)
                            debug_video_frame_count = target_frame_count

            except Exception:
                log.exception("OpenCV failed for camera=%s", self.camera.code)

            finally:
                if capture is not None:
                    capture.release()

            if not self._stop_event.is_set():
                self._stop_event.wait(RECONNECT_DELAY_S)

        if debug_writer is not None:
            debug_writer.release()
            log.info(
                "Debug video saved camera=%s path=%s frames=%d",
                self.camera.code,
                debug_video_path,
                debug_video_frame_count,
            )
        log.info("Worker camera=%s has stopped", self.camera.code)
