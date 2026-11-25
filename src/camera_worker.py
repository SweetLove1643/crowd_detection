# Bước 3: 1 camera = 1 tiến trình FFmpeg đọc RTSP + 1 thread xử lý AI
import json
import logging
import math
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import torch

from src.analyze import ProximityEvaluator
from src.camera_manager import Camera
import src.config as cf
from src.detector import PersonDetector
from src.publisher import publish_bbox, publish_crowd_event

log = logging.getLogger("camera_worker")

DEBUG_FRAME_SAVE_INTERVAL_S = 5.0
PERFORMANCE_LOG_INTERVAL_S = 5.0
DEBUG_VIDEO_FPS = cf.DEBUG_VIDEO_FPS
CROWD_TRACK_GRACE_S = cf.CROWD_TRACK_GRACE_SECONDS
CROWD_EVENT_RETRY_DELAY_S = 5.0
PERSON_COLOR_BGR = (0, 255, 0)
CROWD_COLOR_BGR = (0, 0, 255)
PERSON_COLOR_HEX = "#00ff00"
CROWD_COLOR_HEX = "#ff0000"
PERFORMANCE_STAGES = (
    "read",
    "infer",
    "cluster",
    "crowd",
    "bbox_build",
    "bbox_publish",
    "annotate",
    "snapshot_encode",
    "snapshot_write",
    "video_write",
    "loop",
)

Detection = tuple[float, float, float, float, float, int | None]
Cluster = list[int]
CrowdedCluster = tuple[int, Cluster]
BBoxCoords = tuple[float, float, float, float]


def _cluster_bbox(
    cluster: Cluster,
    detections: list[Detection],
) -> BBoxCoords | None:
    if not cluster:
        return None
    boxes = [detections[i] for i in cluster if 0 <= i < len(detections)]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (float(x1), float(y1), float(x2), float(y2))


def _bbox_iou(box_a: BBoxCoords, box_b: BBoxCoords) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _bbox_center_distance_ratio(
    box_a: BBoxCoords,
    box_b: BBoxCoords,
) -> float:
    acx = (box_a[0] + box_a[2]) * 0.5
    acy = (box_a[1] + box_a[3]) * 0.5
    bcx = (box_b[0] + box_b[2]) * 0.5
    bcy = (box_b[1] + box_b[3]) * 0.5
    dist = math.hypot(acx - bcx, acy - bcy)
    size_a = max(box_a[2] - box_a[0], box_a[3] - box_a[1], 1.0)
    size_b = max(box_b[2] - box_b[0], box_b[3] - box_b[1], 1.0)
    ref_size = (size_a + size_b) * 0.5
    return dist / ref_size


def _prune_old_files(directory: Path, pattern: str, max_files: int) -> None:
    if max_files <= 0 or not directory.is_dir():
        return
    try:
        files = sorted(
            directory.glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(files) - max_files
        for old_file in files[: max(0, excess)]:
            try:
                old_file.unlink(missing_ok=True)
            except OSError:
                pass
    except OSError:
        pass


def _smooth_bbox(
    prev_box: BBoxCoords | None,
    new_box: BBoxCoords,
    alpha: float,
) -> BBoxCoords:
    if prev_box is None or alpha >= 1.0:
        return new_box
    inv = 1.0 - alpha
    return (
        prev_box[0] * inv + new_box[0] * alpha,
        prev_box[1] * inv + new_box[1] * alpha,
        prev_box[2] * inv + new_box[2] * alpha,
        prev_box[3] * inv + new_box[3] * alpha,
    )


@dataclass(frozen=True)
class DisplayCrowd:
    crowd_id: int
    bbox: BBoxCoords
    person_count: int
    member_indices: tuple[int, ...]
    track_ids: tuple[int, ...]
    holding: bool


@dataclass
class _CrowdCandidate:
    crowd_id: int
    track_ids: set[int]
    started_at: float
    last_seen: float
    last_crowded_at: float
    person_count: int
    bbox: BBoxCoords | None = None
    smoothed_bbox: BBoxCoords | None = None
    history_track_ids: set[int] = field(default_factory=set)
    member_indices: list[int] = field(default_factory=list)
    notified: bool = False
    publishing: bool = False
    last_publish_attempt: float = 0.0
    last_successful_publish: float | None = None

    def __post_init__(self) -> None:
        if not self.history_track_ids and self.track_ids:
            self.history_track_ids = set(self.track_ids)
        if self.smoothed_bbox is None and self.bbox is not None:
            self.smoothed_bbox = self.bbox


class CrowdPersistence:
    def __init__(
        self,
        duration_s: float,
        grace_s: float = CROWD_TRACK_GRACE_S,
        hold_s: float | None = None,
        smoothing_alpha: float | None = None,
    ) -> None:
        self.duration_s = float(duration_s)
        self.grace_s = float(grace_s)
        self.hold_s = (
            float(cf.CROWD_BBOX_HOLD_SECONDS)
            if hold_s is None
            else float(hold_s)
        )
        self.smoothing_alpha = (
            float(cf.CROWD_BBOX_SMOOTHING_ALPHA)
            if smoothing_alpha is None
            else float(smoothing_alpha)
        )
        self._candidates: list[_CrowdCandidate] = []
        self._next_crowd_id: int = 1

    @staticmethod
    def _real_ids(ids: set[int]) -> set[int]:
        return {t for t in ids if t >= 0}

    def update(
        self,
        clusters: list[Cluster],
        crowded_clusters: list[CrowdedCluster],
        detections: list[Detection],
        now: float,
        frame_interval_s: float = 0.0,
    ) -> tuple[list[DisplayCrowd], list[tuple[int, Cluster, set[int]]]]:
        effective_hold_s = (
            max(self.hold_s, frame_interval_s * 2.5)
            if self.hold_s > 0.0
            else 0.0
        )
        effective_event_grace_s = max(
            self.grace_s,
            effective_hold_s,
            frame_interval_s * 2.5,
        )

        self._candidates = [
            candidate
            for candidate in self._candidates
            if now - candidate.last_crowded_at <= effective_event_grace_s
        ]

        for candidate in self._candidates:
            candidate.member_indices = []

        matched_crowd_ids: set[int] = set()
        claimed_cluster_indices: set[int] = {
            cluster_index for cluster_index, _ in crowded_clusters
        }
        ready: list[tuple[int, list[int], set[int]]] = []

        for cluster_index, cluster in crowded_clusters:
            tracked_ids = {
                int(detections[index][5])
                for index in cluster
                if detections[index][5] is not None
            }
            synthetic_ids = {
                -(cluster_index * 1000 + pos + 1)
                for pos in range(len(cluster))
            }
            effective_ids = tracked_ids if tracked_ids else synthetic_ids
            real_effective_ids = self._real_ids(effective_ids)
            cluster_box = _cluster_bbox(cluster, detections)

            match_index = None
            best_score = -1.0

            for index, cand in enumerate(self._candidates):
                if cand.crowd_id in matched_crowd_ids:
                    continue
                overlap = len(self._real_ids(cand.track_ids) & real_effective_ids)
                hist_overlap = len(
                    self._real_ids(cand.history_track_ids) & real_effective_ids
                )

                iou = (
                    _bbox_iou(cand.bbox, cluster_box)
                    if cand.bbox is not None and cluster_box is not None
                    else 0.0
                )
                center_ratio = (
                    _bbox_center_distance_ratio(cand.bbox, cluster_box)
                    if cand.bbox is not None and cluster_box is not None
                    else float("inf")
                )
                spatial_match = (
                    iou >= cf.CROWD_SPATIAL_IOU_THRESHOLD
                    or center_ratio <= cf.CROWD_CENTER_DISTANCE_RATIO
                )

                if overlap > 0 or hist_overlap > 0 or spatial_match:
                    score = (
                        overlap * 10.0
                        + hist_overlap * 3.0
                        + iou * 5.0
                        + max(0.0, 1.0 - center_ratio)
                    )
                    if score > best_score:
                        best_score = score
                        match_index = index

            if match_index is not None:
                candidate = self._candidates[match_index]
                matched_crowd_ids.add(candidate.crowd_id)
                candidate.track_ids = set(effective_ids)
                candidate.history_track_ids |= effective_ids
                if len(candidate.history_track_ids) > 64:
                    candidate.history_track_ids = set(effective_ids)
                candidate.person_count = len(cluster)
                candidate.member_indices = list(cluster)
                candidate.last_seen = now
                candidate.last_crowded_at = now
                if cluster_box is not None:
                    candidate.bbox = cluster_box
                    candidate.smoothed_bbox = _smooth_bbox(
                        candidate.smoothed_bbox,
                        cluster_box,
                        self.smoothing_alpha,
                    )
            else:
                crowd_id = self._next_crowd_id
                self._next_crowd_id += 1
                candidate = _CrowdCandidate(
                    crowd_id=crowd_id,
                    track_ids=set(effective_ids),
                    started_at=now,
                    last_seen=now,
                    last_crowded_at=now,
                    person_count=len(cluster),
                    bbox=cluster_box,
                    smoothed_bbox=cluster_box,
                    history_track_ids=set(effective_ids),
                    member_indices=list(cluster),
                )
                self._candidates.append(candidate)
                matched_crowd_ids.add(crowd_id)

            persistence_met = now - candidate.started_at >= self.duration_s
            retry_due = (
                now - candidate.last_publish_attempt >= CROWD_EVENT_RETRY_DELAY_S
            )
            last_successful_publish = candidate.last_successful_publish
            if candidate.notified and last_successful_publish is not None:
                repeat_interval_elapsed = (
                    now - last_successful_publish
                    >= cf.CROWD_EVENT_REPEAT_INTERVAL_SECONDS
                )
                previous_attempt_failed = (
                    candidate.last_publish_attempt > last_successful_publish
                )
                publish_due = repeat_interval_elapsed and (
                    not previous_attempt_failed or retry_due
                )
            else:
                publish_due = not candidate.notified and (
                    candidate.last_publish_attempt == 0 or retry_due
                )

            if (
                persistence_met
                and not candidate.publishing
                and publish_due
            ):
                candidate.publishing = True
                candidate.last_publish_attempt = now
                ready.append((candidate.crowd_id, cluster, set(effective_ids)))

        matched_candidates = [
            c
            for c in self._candidates
            if c.crowd_id in matched_crowd_ids
        ]
        absorbed_ids: set[int] = set()

        for candidate in self._candidates:
            if candidate.crowd_id in matched_crowd_ids:
                continue

            cand_real = self._real_ids(candidate.track_ids)
            for other in matched_candidates:
                other_real = self._real_ids(other.track_ids)
                id_merged = bool(cand_real) and (
                    len(cand_real & other_real) >= max(2, len(cand_real) // 2)
                )
                box_merged = (
                    candidate.bbox is not None
                    and other.bbox is not None
                    and _bbox_iou(candidate.bbox, other.bbox) >= 0.35
                )
                if id_merged or box_merged:
                    other.history_track_ids |= candidate.history_track_ids
                    if len(other.history_track_ids) > 64:
                        other.history_track_ids = set(other.track_ids)
                    other.started_at = min(other.started_at, candidate.started_at)
                    absorbed_ids.add(candidate.crowd_id)
                    break

            if candidate.crowd_id in absorbed_ids:
                continue

            if now - candidate.last_crowded_at <= effective_hold_s:
                best_cluster_index = None
                best_cluster: Cluster | None = None
                best_sub_box: BBoxCoords | None = None
                best_sub_ids: set[int] = set()
                best_score = -1.0

                for cluster_index, cluster in enumerate(clusters):
                    if (
                        cluster_index in claimed_cluster_indices
                        or len(cluster) < 2
                    ):
                        continue
                    sub_ids = {
                        int(detections[index][5])
                        for index in cluster
                        if detections[index][5] is not None
                    }
                    sub_box = _cluster_bbox(cluster, detections)
                    overlap = len(cand_real & sub_ids)
                    hist_overlap = len(
                        self._real_ids(candidate.history_track_ids) & sub_ids
                    )
                    iou = (
                        _bbox_iou(candidate.bbox, sub_box)
                        if candidate.bbox is not None and sub_box is not None
                        else 0.0
                    )
                    center_ratio = (
                        _bbox_center_distance_ratio(candidate.bbox, sub_box)
                        if candidate.bbox is not None and sub_box is not None
                        else float("inf")
                    )
                    spatial_match = (
                        iou >= cf.CROWD_SPATIAL_IOU_THRESHOLD
                        or center_ratio <= cf.CROWD_CENTER_DISTANCE_RATIO
                    )
                    if overlap > 0 or hist_overlap > 0 or spatial_match:
                        score = (
                            overlap * 10.0
                            + hist_overlap * 3.0
                            + iou * 5.0
                            + max(0.0, 1.0 - center_ratio)
                        )
                        if score > best_score:
                            best_score = score
                            best_cluster_index = cluster_index
                            best_cluster = cluster
                            best_sub_box = sub_box
                            best_sub_ids = sub_ids

                if best_cluster_index is not None and best_cluster is not None:
                    claimed_cluster_indices.add(best_cluster_index)
                    candidate.member_indices = list(best_cluster)
                    candidate.last_seen = now
                    if best_sub_ids:
                        candidate.history_track_ids |= best_sub_ids
                        if len(candidate.history_track_ids) > 64:
                            candidate.history_track_ids = set(best_sub_ids)
                    if best_sub_box is not None:
                        candidate.bbox = best_sub_box
                        candidate.smoothed_bbox = _smooth_bbox(
                            candidate.smoothed_bbox,
                            best_sub_box,
                            self.smoothing_alpha,
                        )

        if absorbed_ids:
            self._candidates = [
                candidate
                for candidate in self._candidates
                if candidate.crowd_id not in absorbed_ids
            ]

        display_crowds: list[DisplayCrowd] = []
        for candidate in self._candidates:
            if now - candidate.last_crowded_at > effective_hold_s:
                continue
            box = candidate.smoothed_bbox or candidate.bbox
            if box is None:
                continue
            is_holding = candidate.crowd_id not in matched_crowd_ids
            if candidate.member_indices:
                visible_track_ids = {
                    int(detections[index][5])
                    for index in candidate.member_indices
                    if 0 <= index < len(detections)
                    and detections[index][5] is not None
                }
            else:
                visible_track_ids = self._real_ids(candidate.track_ids)
            display_crowds.append(
                DisplayCrowd(
                    crowd_id=candidate.crowd_id,
                    bbox=box,
                    person_count=candidate.person_count,
                    member_indices=tuple(candidate.member_indices),
                    track_ids=tuple(sorted(visible_track_ids)),
                    holding=is_holding,
                )
            )

        return display_crowds, ready

    def complete_publish(
        self,
        track_ids: set[int],
        success: bool,
        completed_at: float,
    ) -> None:
        for candidate in self._candidates:
            if (
                candidate.track_ids & track_ids
                or candidate.history_track_ids & track_ids
            ):
                candidate.publishing = False
                if success:
                    candidate.notified = True
                    candidate.last_successful_publish = completed_at

    def has_active_candidates(self) -> bool:
        return bool(self._candidates)

    def max_elapsed(self, now: float) -> float:
        if not self._candidates:
            return 0.0
        return max(max(0.0, now - c.started_at) for c in self._candidates)


def probe_stream(camera: Camera) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            "4000000",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            camera.rtsp_url,
        ],
        capture_output=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFprobe could not open camera stream camera={camera.code}"
        )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found for camera={camera.code}")
    return streams[0]


class FFmpegFrameGrabber:
    """Đọc RTSP bằng tiến trình FFmpeg độc lập, luôn giữ frame BGR mới nhất (tránh trễ buffer)."""

    def __init__(
        self,
        camera: Camera,
        stop_event: threading.Event,
        *,
        width: int | None = None,
        height: int | None = None,
        max_width: int | None = None,
        startup_timeout: float = 10.0,
        stall_timeout: float = 3.5,
        max_frame_age: float = 0.40,
        reconnect_delay: float = 0.5,
        decoder: str | None = None,
    ) -> None:
        if (width is not None and width <= 0) or (
            height is not None and height <= 0
        ):
            raise ValueError("Kích thước frame phải dương")
        resolved_max_width = (
            int(cf.FFMPEG_MAX_WIDTH) if max_width is None else int(max_width)
        )
        if resolved_max_width < 0:
            raise ValueError("max_width không được âm")
        self.camera = camera
        self.width = width
        self.height = height
        self._max_width = resolved_max_width
        self._needs_scale: bool = bool(width and height)
        self._stop_event = stop_event
        self._closed = threading.Event()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_seq: int = 0
        self._latest_read_ms: float = 0.0
        self._last_frame_at: float | None = None
        self._session_progress_at: float | None = None
        self._session_has_frames: bool = False
        self._connected: bool = False
        self._startup_timeout = startup_timeout
        self._stall_timeout = stall_timeout
        self._max_frame_age = max_frame_age
        self._reconnect_delay = reconnect_delay
        self._decoder = decoder
        self.sessions = 0
        self.watchdog_restarts = 0
        self.reference_errors = 0
        self.decoder_errors = 0

    def _stopped(self) -> bool:
        return self._closed.is_set() or self._stop_event.is_set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"rtsp-{self.camera.code}",
            daemon=True,
        )
        self._watchdog = threading.Thread(
            target=self._watch,
            name=f"rtsp-watch-{self.camera.code}",
            daemon=True,
        )
        self._thread.start()
        self._watchdog.start()

    def stop(self, timeout: float | None = 0.5) -> None:
        self._closed.set()
        with self._cond:
            self._kill_process()
            self._cond.notify_all()
        for thread in (self._thread, self._watchdog):
            if thread is not None:
                thread.join(timeout=timeout)

    def _fresh(self, last_seq: int) -> bool:
        return (
            self._latest_seq > last_seq
            and self._latest_frame is not None
            and self._last_frame_at is not None
            and time.monotonic() - self._last_frame_at <= self._max_frame_age
        )

    def get_latest_frame(
        self,
        last_seq: int,
        timeout: float = 0.2,
    ) -> tuple[int, float, np.ndarray | None]:
        began = time.perf_counter()
        with self._cond:
            self._cond.wait_for(
                lambda: self._stopped() or self._fresh(last_seq),
                timeout=timeout,
            )
            if self._stopped() or not self._fresh(last_seq):
                return last_seq, 0.0, None
            return (
                self._latest_seq,
                (time.perf_counter() - began) * 1000,
                self._latest_frame,
            )

    def peek_newer_frame(
        self,
        current_seq: int,
    ) -> tuple[int, np.ndarray | None]:
        with self._cond:
            if not self._stopped() and self._fresh(current_seq):
                return self._latest_seq, self._latest_frame
            return current_seq, None

    def _kill_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()

    def _watch(self) -> None:
        while not self._closed.wait(0.10):
            with self._cond:
                if self._stopped():
                    self._kill_process()
                    self._cond.notify_all()
                    return
                if (
                    self._process is None
                    or self._process.poll() is not None
                    or self._session_progress_at is None
                ):
                    continue
                limit = (
                    self._stall_timeout
                    if self._session_has_frames
                    else self._startup_timeout
                )
                if time.monotonic() - self._session_progress_at >= limit:
                    self.watchdog_restarts += 1
                    log.warning(
                        "FFmpeg watchdog restarting stalled stream camera=%s "
                        "restarts=%d has_frames=%s",
                        self.camera.code,
                        self.watchdog_restarts,
                        self._session_has_frames,
                    )
                    self._kill_process()

    def _drain_errors(self, pipe) -> None:
        for line in pipe:
            if (
                b"Could not find ref with POC" in line
                or b"Error constructing the frame RPS" in line
            ):
                self.reference_errors += 1
            if any(
                term in line
                for term in (
                    b"Cannot load lib",
                    b"Error while opening decoder",
                    b"CUDA_ERROR",
                )
            ):
                self.decoder_errors += 1

    def _command(self) -> list[str]:
        decoder = ["-c:v", self._decoder] if self._decoder else []
        scale_filter = (
            ["-vf", f"scale={self.width}:{self.height}:flags=fast_bilinear"]
            if self._needs_scale
            else []
        )
        return [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            "4000000",
            "-threads",
            "2",
            *decoder,
            "-i",
            self.camera.rtsp_url,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            *scale_filter,
            "-vsync",
            "0",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-threads:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

    def _store_frame(self, frame: np.ndarray, read_ms: float) -> None:
        with self._cond:
            now = time.monotonic()
            self._latest_frame = frame
            self._latest_seq += 1
            self._latest_read_ms = read_ms
            self._last_frame_at = self._session_progress_at = now
            self._session_has_frames = self._connected = True
            self._cond.notify_all()

    def _ensure_dimensions(self) -> bool:
        if self.width and self.height:
            return True
        try:
            stream = probe_stream(self.camera)
            src_w = int(stream["width"])
            src_h = int(stream["height"])
            if src_w <= 0 or src_h <= 0:
                raise ValueError(f"Invalid stream size {src_w}x{src_h}")
            if self._max_width > 0 and src_w > self._max_width:
                target_w = max(2, (self._max_width // 2) * 2)
                target_h = max(
                    2, (int(round(src_h * (target_w / src_w))) // 2) * 2
                )
                self.width = target_w
                self.height = target_h
                self._needs_scale = True
            else:
                self.width = src_w
                self.height = src_h
                self._needs_scale = False
            log.info(
                "Opened RTSP stream info camera=%s src=%dx%d out=%dx%d codec=%s fps=%s",
                self.camera.code,
                src_w,
                src_h,
                self.width,
                self.height,
                stream.get("codec_name"),
                stream.get("avg_frame_rate"),
            )
            return True
        except Exception as exc:
            log.warning(
                "Cannot probe RTSP camera=%s: %s",
                self.camera.code,
                exc,
            )
            return False

    def _read_loop(self) -> None:
        while not self._stopped():
            if not self._ensure_dimensions():
                self._closed.wait(self._reconnect_delay)
                continue

            assert self.width is not None and self.height is not None
            process = subprocess.Popen(
                self._command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with self._cond:
                self._process = process
                self.sessions += 1
                self._session_progress_at = time.monotonic()
                self._session_has_frames = False
                if self._stopped():
                    self._kill_process()
            stderr_reader = threading.Thread(
                target=self._drain_errors,
                args=(process.stderr,),
                daemon=True,
            )
            stderr_reader.start()
            try:
                while not self._stopped() and process.stdout is not None:
                    began = time.perf_counter()
                    frame = np.empty(
                        (self.height, self.width, 3),
                        dtype=np.uint8,
                    )
                    count = process.stdout.readinto(
                        memoryview(frame).cast("B")
                    )
                    if count != frame.nbytes:
                        break
                    self._store_frame(
                        frame,
                        (time.perf_counter() - began) * 1000,
                    )
            finally:
                with self._cond:
                    self._kill_process()
                    self._process = None
                    self._latest_frame = None
                    self._connected = False
                    self._cond.notify_all()
                process.wait()
                if process.stdout is not None:
                    process.stdout.close()
                stderr_reader.join(1)
                if process.stderr is not None:
                    process.stderr.close()
            self._closed.wait(self._reconnect_delay)


def _build_bbox_detections(
    detections: list[Detection],
    clusters: list[Cluster],
    display_crowds: list[DisplayCrowd],
    frame_width: int,
    frame_height: int,
    publish_all_person_bbox: bool,
    padding_percent: float,
) -> list[dict[str, object]]:
    detection_clusters = {
        detection_index: cluster_index + 1
        for cluster_index, cluster in enumerate(clusters)
        for detection_index in cluster
    }
    crowd_detection_indices = {
        detection_index
        for crowd in display_crowds
        for detection_index in crowd.member_indices
    }
    bbox_detections: list[dict[str, object]] = []

    for detection_index, detection in enumerate(detections):
        if (
            detection_index in crowd_detection_indices
            or not publish_all_person_bbox
        ):
            continue

        x1, y1, x2, y2, confidence, track_id = detection
        person_id = (
            f"Người_{track_id}"
            if track_id is not None
            else f"Người_{detection_index}"
        )
        person_bbox: dict[str, object] = {
            "id": person_id,
            "label": f"Người",
            # "label": f"{person_id} {confidence:.2f}",
            "class": "person",
            "confidence": round(confidence, 4),
            "bbox": [
                round(x1 / frame_width, 6),
                round(y1 / frame_height, 6),
                round(x2 / frame_width, 6),
                round(y2 / frame_height, 6),
            ],
            "color": PERSON_COLOR_HEX,
            "cluster_id": detection_clusters[detection_index],
        }
        if track_id is not None:
            person_bbox["track_id"] = track_id
        bbox_detections.append(person_bbox)

    padding = padding_percent / 100
    for crowd in display_crowds:
        x1, y1, x2, y2 = crowd.bbox
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_width, x2 + pad_x)
        y2 = min(frame_height, y2 + pad_y)
        bbox_detections.append(
            {
                "id": f"crowd_{crowd.crowd_id}",
                "label": f"{crowd.person_count} người",
                # "label": f"Cụm {crowd.crowd_id} ({crowd.person_count} người)",
                "class": "crowd",
                "bbox": [
                    round(x1 / frame_width, 6),
                    round(y1 / frame_height, 6),
                    round(x2 / frame_width, 6),
                    round(y2 / frame_height, 6),
                ],
                "color": CROWD_COLOR_HEX,
                "cluster_id": crowd.crowd_id,
                "person_count": crowd.person_count,
                "track_ids": list(crowd.track_ids),
            }
        )

    return bbox_detections


def _annotate_bbox_detections(
    frame: np.ndarray,
    detections: list[Detection],
    display_crowds: list[DisplayCrowd],
    padding_percent: float,
    inference_ms: float,
    processing_fps: float,
) -> np.ndarray:
    annotated = frame.copy()

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
            0.9,
            color,
            3,
            cv2.LINE_AA,
        )

    padding = padding_percent / 100
    for crowd in display_crowds:
        x1, y1, x2, y2 = crowd.bbox
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        draw_box(
            (
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(frame.shape[1], x2 + pad_x),
                min(frame.shape[0], y2 + pad_y),
            ),
            CROWD_COLOR_BGR,
            f"{crowd.person_count} người",
        )

    return annotated


def _frame_description(frame: np.ndarray) -> str:
    return (
        f"shape={frame.shape} dtype={frame.dtype} strides={frame.strides} "
        f"contiguous={frame.flags.c_contiguous}"
    )


def _validate_debug_frame(frame: np.ndarray, operation: str) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            f"Unsupported frame for {operation}: {_frame_description(frame)}"
        )
    return np.ascontiguousarray(frame)


def _publish_crowd_event_async(
    client: mqtt.Client,
    camera_id: str,
    camera_code: str,
    clusters: list[dict],
    snapshot_jpg: bytes | None,
    track_ids: set[int],
    result_queue: queue.Queue[tuple[set[int], bool]],
) -> bool:
    def publish_and_report() -> None:
        publish_started = time.perf_counter()
        try:
            success = publish_crowd_event(
                client,
                camera_id,
                camera_code,
                clusters,
                jpg=snapshot_jpg,
            )
        except Exception:
            log.exception("Crowd event publisher crashed camera=%s", camera_code)
            success = False
        log.info(
            "Crowd event delivery camera=%s success=%s duration_ms=%.0f",
            camera_code,
            success,
            (time.perf_counter() - publish_started) * 1000,
        )
        result_queue.put((track_ids, success))

    thread = threading.Thread(
        target=publish_and_report,
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

    def signal_stop(self) -> None:
        """Gửi tín hiệu dừng không chặn (non-blocking) cho worker."""
        self._stop_event.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"Woker for camera {self.camera.code} started")

        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.camera.code}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = 7) -> None:
        self._stop_event.set()

        if self._thread is None:
            return

        self._thread.join(timeout=timeout)

        if timeout is not None and self._thread.is_alive():
            log.warning(
                "Worker camera=%s haven't stopped yet %.1f second",
                self.camera.code,
                timeout,
            )

    def _run(self) -> None:
        cv2.setNumThreads(1)
        weight_path = Path(cf.WEIGHT_YOLO_MODEL_PATH)
        confidence = float(cf.YOLO_MODEL_CONF)
        grabber: FFmpegFrameGrabber | None = None
        debug_writer = None
        debug_video_frame_count = 0
        debug_video_disabled = False
        debug_video_path: Path | None = None

        try:
            detector = PersonDetector(
                weights_path=weight_path,
                confidence=confidence,
            )
            analyzer = ProximityEvaluator()
            crowd_persistence = CrowdPersistence(cf.CROWD_PERSISTENCE_SECONDS)
            grabber = FFmpegFrameGrabber(self.camera, self._stop_event)
            grabber.start()

            was_crowded = False
            last_cluster_log_time = 0.0
            last_snapshot_time = 0.0
            last_performance_log_time = time.monotonic()
            last_frame_time: float | None = None
            last_seq = 0
            frame_count = 0
            bbox_publish_count = 0
            bbox_publish_success_count = 0
            stage_totals_ms = {stage: 0.0 for stage in PERFORMANCE_STAGES}
            stage_max_ms = {stage: 0.0 for stage in PERFORMANCE_STAGES}
            stage_counts = {stage: 0 for stage in PERFORMANCE_STAGES}
            publish_results: queue.Queue[tuple[set[int], bool]] = queue.Queue()

            image_save_dir = (
                Path(__file__).resolve().parents[1] / "resources" / "img_test"
            )
            debug_video_dir = (
                Path(__file__).resolve().parents[1]
                / "resources"
                / "debug_videos"
            )
            if cf.DEBUG_FRAME_SAVE_ENABLED:
                image_save_dir.mkdir(parents=True, exist_ok=True)
                _prune_old_files(
                    image_save_dir, "*.jpg", cf.DEBUG_MAX_SAVED_FRAMES
                )
            if cf.DEBUG_VIDEO_ENABLED:
                debug_video_dir.mkdir(parents=True, exist_ok=True)
                _prune_old_files(
                    debug_video_dir, "*.mp4", cf.DEBUG_MAX_SAVED_VIDEOS
                )

            log.info(
                "Processing options camera=%s cpu_threads=%d debug_video=%s "
                "debug_snapshots=%s",
                self.camera.code,
                torch.get_num_threads(),
                cf.DEBUG_VIDEO_ENABLED,
                cf.DEBUG_FRAME_SAVE_ENABLED,
            )

            while not self._stop_event.is_set():
                try:
                    while not self._stop_event.is_set():
                        frame_loop_started = time.perf_counter()
                        seq, read_ms, frame = grabber.get_latest_frame(
                            last_seq,
                            timeout=0.5,
                        )
                        if frame is None or seq == last_seq:
                            continue
                        last_seq = seq

                        if (
                            cf.DEBUG_VIDEO_ENABLED
                            and debug_writer is None
                            and not debug_video_disabled
                        ):
                            safe_camera_code = "".join(
                                char if char.isalnum() or char in "-_" else "_"
                                for char in self.camera.code
                            )
                            timestamp = datetime.now().strftime(
                                "%Y%m%d_%H%M%S_%f"
                            )[:-3]
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
                                log.info(
                                    "Debug video started camera=%s path=%s "
                                    "fps=%.1f size=%dx%d codec=mp4v",
                                    self.camera.code,
                                    debug_video_path,
                                    DEBUG_VIDEO_FPS,
                                    frame_width,
                                    frame_height,
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
                        frame_interval_s = (
                            frame_started_at - last_frame_time
                            if last_frame_time is not None
                            and frame_started_at > last_frame_time
                            else 0.0
                        )
                        processing_fps = (
                            1.0 / frame_interval_s
                            if frame_interval_s > 0.0
                            else 0.0
                        )
                        last_frame_time = frame_started_at

                        inference_started = time.perf_counter()
                        if self._stop_event.is_set():
                            break

                        def _refresh_latest_frame() -> np.ndarray | None:
                            nonlocal last_seq, frame
                            newer_seq, newer_frame = grabber.peek_newer_frame(
                                last_seq
                            )
                            if newer_frame is not None and newer_seq > last_seq:
                                last_seq = newer_seq
                                frame = newer_frame
                                return newer_frame
                            return None

                        dets = detector.track(
                            frame,
                            latest_frame_getter=_refresh_latest_frame,
                        )
                        if self._stop_event.is_set():
                            break
                        inference_ms = (
                            time.perf_counter() - inference_started
                        ) * 1000

                        cluster_started = time.perf_counter()
                        clusters = analyzer.clusters(dets)
                        crowded_clusters = [
                            (cluster_index, cluster)
                            for cluster_index, cluster in enumerate(clusters)
                            if len(cluster) >= cf.NUMBER_PERSON_IS_CROWDED
                        ]
                        cluster_ms = (
                            time.perf_counter() - cluster_started
                        ) * 1000

                        now = time.monotonic()
                        crowd_started = time.perf_counter()
                        while True:
                            try:
                                published_track_ids, publish_succeeded = (
                                    publish_results.get_nowait()
                                )
                            except queue.Empty:
                                break
                            crowd_persistence.complete_publish(
                                published_track_ids,
                                publish_succeeded,
                                now,
                            )
                        display_crowds, ready_clusters = crowd_persistence.update(
                            clusters,
                            crowded_clusters,
                            dets,
                            now,
                            frame_interval_s=frame_interval_s,
                        )
                        is_crowded = bool(display_crowds)
                        crowd_ms = (time.perf_counter() - crowd_started) * 1000

                        frame_height, frame_width = frame.shape[:2]
                        bbox_build_started = time.perf_counter()
                        bbox_detections = _build_bbox_detections(
                            dets,
                            clusters,
                            display_crowds,
                            frame_width,
                            frame_height,
                            cf.PUBLISH_ALL_PERSON_BBOX,
                            cf.CROWD_BBOX_PADDING_PERCENT,
                        )
                        bbox_build_ms = (
                            time.perf_counter() - bbox_build_started
                        ) * 1000

                        hud_text = (
                            f"Số đám đông: {len(display_crowds)} cụm"
                            if is_crowded
                            else f"Số người phát hiện: {len(dets)}"
                        )
                        hud_color = (
                            CROWD_COLOR_HEX if is_crowded else PERSON_COLOR_HEX
                        )
                        bbox_publish_started = time.perf_counter()
                        bbox_published = publish_bbox(
                            self.client,
                            self.camera.id,
                            self.camera.code,
                            bbox_detections,
                            hud_text,
                            hud_color,
                        )
                        bbox_publish_ms = (
                            time.perf_counter() - bbox_publish_started
                        ) * 1000

                        frame_count += 1
                        bbox_publish_count += 1
                        bbox_publish_success_count += int(bbox_published)

                        if now - last_cluster_log_time >= 1.0:
                            cluster_details = [
                                {
                                    "cluster": cluster_index + 1,
                                    "people": len(cluster),
                                    "detections": cluster,
                                    "crowd": len(cluster)
                                    >= cf.NUMBER_PERSON_IS_CROWDED,
                                    "track_ids": [
                                        dets[index][5]
                                        for index in cluster
                                        if dets[index][5] is not None
                                    ],
                                }
                                for cluster_index, cluster in enumerate(
                                    clusters
                                )
                            ]
                            log.info(
                                "Cluster summary camera=%s people=%d clusters=%d "
                                "display_crowds=%d crowd_threshold>=%d details=%s",
                                self.camera.code,
                                len(dets),
                                len(clusters),
                                len(display_crowds),
                                cf.NUMBER_PERSON_IS_CROWDED,
                                cluster_details,
                            )
                            last_cluster_log_time = now

                        has_candidates = (
                            crowd_persistence.has_active_candidates()
                        )
                        if is_crowded and not was_crowded:
                            log.info(
                                "Crowd candidate detected camera=%s active_crowds=%d; waiting %.1fs persistence",
                                self.camera.code,
                                len(display_crowds),
                                cf.CROWD_PERSISTENCE_SECONDS,
                            )
                            was_crowded = True
                        elif was_crowded and not has_candidates:
                            log.info("Crowd cleared camera=%s", self.camera.code)
                            was_crowded = False

                        snapshot_due = (
                            cf.DEBUG_FRAME_SAVE_ENABLED
                            and now - last_snapshot_time
                            >= DEBUG_FRAME_SAVE_INTERVAL_S
                        )
                        needs_annotated_frame = (
                            cf.DEBUG_VIDEO_ENABLED
                            or snapshot_due
                            or bool(ready_clusters)
                        )
                        annotate_ms = 0.0
                        debug_frame = None
                        if needs_annotated_frame:
                            annotate_started = time.perf_counter()
                            debug_frame = _annotate_bbox_detections(
                                frame,
                                dets,
                                display_crowds,
                                cf.CROWD_BBOX_PADDING_PERCENT,
                                inference_ms,
                                processing_fps,
                            )
                            debug_frame = _validate_debug_frame(
                                debug_frame,
                                "debug output",
                            )
                            annotate_ms = (
                                time.perf_counter() - annotate_started
                            ) * 1000

                        snapshot_bytes = None
                        snapshot_encode_ms = 0.0
                        if snapshot_due or ready_clusters:
                            snapshot_encode_started = time.perf_counter()
                            try:
                                if debug_frame is None:
                                    raise RuntimeError(
                                        "Annotated frame missing for snapshot"
                                    )
                                encoded, snapshot_buffer = cv2.imencode(
                                    ".jpg",
                                    debug_frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                                )
                                if encoded and snapshot_buffer.size:
                                    snapshot_bytes = snapshot_buffer.tobytes()
                                else:
                                    log.error(
                                        "Failed to encode snapshot camera=%s frame=%s",
                                        self.camera.code,
                                        _frame_description(debug_frame),
                                    )
                            except (cv2.error, RuntimeError):
                                log.exception(
                                    "Failed to encode snapshot camera=%s frame=%s",
                                    self.camera.code,
                                    _frame_description(debug_frame)
                                    if debug_frame is not None
                                    else "missing",
                                )
                            snapshot_encode_ms = (
                                time.perf_counter() - snapshot_encode_started
                            ) * 1000
                            stage_totals_ms["snapshot_encode"] += (
                                snapshot_encode_ms
                            )
                            stage_max_ms["snapshot_encode"] = max(
                                stage_max_ms["snapshot_encode"],
                                snapshot_encode_ms,
                            )
                            stage_counts["snapshot_encode"] += 1

                        if ready_clusters:
                            log.warning(
                                "Persistent crowd detected camera=%s duration=%.1fs clusters=%s",
                                self.camera.code,
                                cf.CROWD_PERSISTENCE_SECONDS,
                                [cluster for _, cluster, _ in ready_clusters],
                            )
                            event_track_ids = set().union(
                                *(
                                    track_ids
                                    for _, _, track_ids in ready_clusters
                                )
                            )
                            event_scheduled = _publish_crowd_event_async(
                                self.client,
                                self.camera.id,
                                self.camera.code,
                                [
                                    {
                                        "cluster_id": crowd_id,
                                        "person_count": len(cluster),
                                        "detections": cluster,
                                        "track_ids": sorted(
                                            t for t in track_ids if t >= 0
                                        ),
                                    }
                                    for crowd_id, cluster, track_ids in ready_clusters
                                ],
                                snapshot_jpg=snapshot_bytes,
                                track_ids=event_track_ids,
                                result_queue=publish_results,
                            )
                            if not event_scheduled:
                                crowd_persistence.complete_publish(
                                    event_track_ids,
                                    False,
                                    now,
                                )
                                log.error(
                                    "Crowd event was not scheduled camera=%s; "
                                    "will retry while crowd persists",
                                    self.camera.code,
                                )
                            else:
                                log.info(
                                    "Crowd event publishing started camera=%s clusters=%d",
                                    self.camera.code,
                                    len(ready_clusters),
                                )

                        if snapshot_due:
                            snapshot_write_started = time.perf_counter()
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                                :-3
                            ]
                            safe_camera_code = "".join(
                                char if char.isalnum() or char in "-_" else "_"
                                for char in self.camera.code
                            )
                            img_path = (
                                image_save_dir / f"{safe_camera_code}_{ts}.jpg"
                            )
                            if snapshot_bytes is None:
                                log.error(
                                    "Skipping debug snapshot because JPEG encoding "
                                    "failed camera=%s path=%s",
                                    self.camera.code,
                                    img_path,
                                )
                            else:
                                try:
                                    img_path.write_bytes(snapshot_bytes)
                                    _prune_old_files(
                                        image_save_dir,
                                        "*.jpg",
                                        cf.DEBUG_MAX_SAVED_FRAMES,
                                    )
                                except OSError:
                                    log.exception(
                                        "Failed to save debug snapshot camera=%s "
                                        "path=%s frame=%s",
                                        self.camera.code,
                                        img_path,
                                        _frame_description(debug_frame)
                                        if debug_frame is not None
                                        else "missing",
                                    )
                            last_snapshot_time = now
                            snapshot_write_ms = (
                                time.perf_counter() - snapshot_write_started
                            ) * 1000
                        else:
                            snapshot_write_ms = 0.0

                        video_write_ms = 0.0
                        if debug_writer is not None and debug_frame is not None:
                            video_write_started = time.perf_counter()
                            try:
                                debug_writer.write(debug_frame)
                                debug_video_frame_count += 1
                            except cv2.error:
                                log.exception(
                                    "Failed to write debug video camera=%s "
                                    "path=%s frame=%s",
                                    self.camera.code,
                                    debug_video_path,
                                    _frame_description(debug_frame),
                                )
                                debug_writer.release()
                                debug_writer = None
                                debug_video_disabled = True
                            video_write_ms = (
                                time.perf_counter() - video_write_started
                            ) * 1000

                        frame_completed_at = time.perf_counter()
                        frame_loop_ms = (
                            frame_completed_at - frame_loop_started
                        ) * 1000
                        frame_stages = {
                            "read": read_ms,
                            "infer": inference_ms,
                            "cluster": cluster_ms,
                            "crowd": crowd_ms,
                            "bbox_build": bbox_build_ms,
                            "annotate": annotate_ms,
                            "bbox_publish": bbox_publish_ms,
                            "snapshot_write": snapshot_write_ms,
                            "video_write": video_write_ms,
                            "loop": frame_loop_ms,
                        }
                        for stage, duration_ms in frame_stages.items():
                            stage_totals_ms[stage] += duration_ms
                            stage_max_ms[stage] = max(
                                stage_max_ms[stage],
                                duration_ms,
                            )
                            stage_counts[stage] += 1

                        elapsed = time.monotonic() - last_performance_log_time
                        if elapsed >= PERFORMANCE_LOG_INTERVAL_S:
                            stage_summary = " ".join(
                                f"{stage}={stage_totals_ms[stage] / stage_counts[stage]:.0f}/"
                                f"{stage_max_ms[stage]:.0f}ms"
                                if stage_counts[stage]
                                else f"{stage}=n/a"
                                for stage in PERFORMANCE_STAGES
                            )
                            log.info(
                                "Pipeline performance camera=%s frames=%d fps=%.2f "
                                "cpu_threads=%d bbox_publish_ok=%d/%d "
                                "stage_avg/max_ms=[%s]",
                                self.camera.code,
                                frame_count,
                                frame_count / elapsed,
                                torch.get_num_threads(),
                                bbox_publish_success_count,
                                bbox_publish_count,
                                stage_summary,
                            )
                            frame_count = 0
                            bbox_publish_count = 0
                            bbox_publish_success_count = 0
                            for stage in PERFORMANCE_STAGES:
                                stage_totals_ms[stage] = 0.0
                                stage_max_ms[stage] = 0.0
                                stage_counts[stage] = 0
                            last_performance_log_time = time.monotonic()
                except Exception:
                    log.exception(
                        "Worker frame error for camera=%s; continuing...",
                        self.camera.code,
                    )
                    self._stop_event.wait(1.0)

        except Exception:
            log.exception("Worker loop failed for camera=%s", self.camera.code)
        finally:
            if grabber is not None:
                grabber.stop()

        if debug_writer is not None:
            debug_writer.release()
        if debug_video_path is not None:
            try:
                video_file_size = (
                    debug_video_path.stat().st_size
                    if debug_video_path.is_file()
                    else 0
                )
                video_check = cv2.VideoCapture(
                    str(debug_video_path),
                    cv2.CAP_FFMPEG,
                )
                video_read_ok, _ = (
                    video_check.read()
                    if video_check.isOpened()
                    else (False, None)
                )
                video_check.release()
            except (OSError, cv2.error):
                log.exception(
                    "Debug video validation failed camera=%s path=%s",
                    self.camera.code,
                    debug_video_path,
                )
                video_file_size = 0
                video_read_ok = False
            if video_file_size <= 0 or not video_read_ok:
                log.error(
                    "Debug video validation failed camera=%s path=%s "
                    "frames_written=%d bytes=%d first_frame_read=%s",
                    self.camera.code,
                    debug_video_path,
                    debug_video_frame_count,
                    video_file_size,
                    video_read_ok,
                )
            else:
                log.info(
                    "Debug video saved and validated camera=%s path=%s "
                    "frames_written=%d bytes=%d",
                    self.camera.code,
                    debug_video_path,
                    debug_video_frame_count,
                    video_file_size,
                )
        log.info("Worker camera=%s has stopped", self.camera.code)
