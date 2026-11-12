# Báo cáo kỹ thuật: Hệ thống AI Crowd Detection (phát hiện đám đông)

> Tài liệu dành cho developer mới. Đọc xong bạn sẽ nắm được hệ thống làm gì, chạy thế nào, code nằm ở đâu, các luồng xử lý ra sao và chỉnh tham số ở chỗ nào.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Công nghệ sử dụng](#2-công-nghệ-sử-dụng)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Kiến trúc tổng thể](#4-kiến-trúc-tổng-thể)
5. [Chi tiết từng module](#5-chi-tiết-từng-module)
6. [Giao tiếp MQTT với VMS](#6-giao-tiếp-mqtt-với-vms)
7. [Cấu hình và điều chỉnh tham số](#7-cấu-hình-và-điều-chỉnh-tham-số)
8. [Cài đặt và chạy](#8-cài-đặt-và-chạy)
9. [Logging, theo dõi hiệu năng, debug](#9-logging-theo-dõi-hiệu-năng-debug)
10. [Các lỗi thường gặp](#10-các-lỗi-thường-gặp)
11. [Bài học từ quá trình tối ưu](#11-bài-học-từ-quá-trình-tối-ưu)

---

## 1. Tổng quan

Đây là một **AI service chạy nền**, tích hợp với hệ thống **Smart VMS** (Video Management System). Service này:

1. **Nhận danh sách camera** từ VMS qua MQTT: camera nào bật module `CROWD_DETECTION` thì service xử lý camera đó.
2. **Đọc luồng RTSP** của từng camera theo thời gian thực.
3. **Phát hiện người** bằng YOLO và **gán ID theo dõi** (tracking) bằng ByteTrack.
4. **Gom cụm** những người đứng gần nhau (theo khoảng cách chân, có bù phối cảnh).
5. **Gửi bbox realtime** (khung người, khung đám đông, dòng HUD) lên VMS để vẽ đè lên video, **mỗi frame một lần**.
6. Nếu một cụm có **≥ N người** và tồn tại **liên tục ≥ T giây**, service **chụp ảnh, upload lên Storage** rồi **gửi sự kiện cảnh báo** (`CROWD_DETECTION`) lên VMS.

Hai loại dữ liệu gửi lên VMS:

| Loại | Tần suất | MQTT QoS | Mục đích |
| :--- | :--- | :--- | :--- |
| **BBox realtime** | Mỗi frame đã inference | 0 (fire-and-forget) | Vẽ khung lên video trên web VMS |
| **Crowd event** | Khi đám đông đủ điều kiện (có lặp lại) | 1 (có ACK từ broker) | Cảnh báo, lưu lịch sử, kèm ảnh |

---

## 2. Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
| :--- | :--- | :--- |
| Ngôn ngữ | Python 3.11 | |
| Phát hiện người | [Ultralytics YOLO](https://docs.ultralytics.com/) (model tự train `final.pt`) | Detect class `person` |
| Tracking | ByteTrack (có sẵn trong Ultralytics, `bytetrack.yaml`) | Gán `track_id` ổn định giữa các frame |
| Runtime suy luận | PyTorch (CPU/CUDA), có hỗ trợ OpenVINO và TensorRT | Chọn tự động theo phần cứng |
| Đọc video | **FFmpeg/FFprobe** chạy tiến trình con (`subprocess`) | Giải mã RTSP (H.264/HEVC) sang raw BGR |
| Xử lý ảnh | OpenCV (`cv2`), NumPy | Vẽ khung, encode JPEG, tính toán ma trận |
| Messaging | MQTT qua **WebSockets** (`paho-mqtt`, CallbackAPIVersion 2) | Nhận cấu hình camera, gửi bbox và event |
| HTTP | `requests` (Session + connection pool) | Upload ảnh snapshot lên Storage |
| Cấu hình | `python-dotenv` + `src/config.py` | `.env` chứa thông tin kết nối, `config.py` chứa tham số AI |

**Yêu cầu hệ thống:** `ffmpeg` và `ffprobe` phải có trong `PATH`. Nếu thiếu, camera sẽ không đọc được frame.

---

## 3. Cấu trúc thư mục

```text
crowded_detection/
├── main.py                  # Entry point: khởi tạo log, kết nối MQTT, chạy vòng lặp
├── .env                     # Thông tin kết nối (MQTT, Zone, Org, Storage) - KHÔNG commit
├── src/
│   ├── config.py            # Nơi duy nhất chứa tham số AI/Crowd/BBox/Debug
│   ├── setup_logging.py     # Cấu hình log (file xoay vòng + console), che mật khẩu trong URL
│   ├── mqtt_client.py       # Tạo MQTT client, subscribe topic camera, quản lý worker
│   ├── camera_manager.py    # Parse danh sách camera, so sánh và start/stop worker
│   ├── camera_worker.py     # Đọc RTSP (FFmpeg), vòng lặp AI, gửi bbox, theo dõi đám đông
│   ├── detector.py          # Bọc YOLO + ByteTrack, chọn backend
│   ├── analyze.py           # ProximityEvaluator: tính lân cận và gom cụm
│   └── publisher.py         # Upload ảnh, publish crowd event và bbox
├── resources/
│   ├── runs/train/          # Kết quả train: weights/ (final.pt, best.pt, last.pt, *_openvino_model), args.yaml, results.csv
│   ├── dataset_yolo/        # Dataset huấn luyện (data.yaml, split)
│   ├── cvat_export/         # Frame xuất từ CVAT để gán nhãn/kiểm thử
│   ├── recordings/          # Video ghi lại từ camera để test
│   ├── img_test/            # (tự tạo) Ảnh debug khi bật DEBUG_FRAME_SAVE_ENABLED
│   └── debug_videos/        # (tự tạo) Video debug khi bật DEBUG_VIDEO_ENABLED
└── logs/
    └── app.log              # Log chạy (xoay vòng 5MB x 3 file)
```

> [!NOTE]
> Repo hiện **không còn `requirements.txt`** (đã bị xóa trong working tree). Xem mục [8. Cài đặt](#8-cài-đặt-và-chạy) để biết các gói cần cài.

---

## 4. Kiến trúc tổng thể

### 4.1. Sơ đồ thành phần

```mermaid
flowchart LR
    VMS["Smart VMS"] -- "danh sách camera (MQTT)" --> MQ["mqtt_client.py"]
    MQ --> CM["CameraManager (reconcile)"]
    CM -- "start/stop" --> W1["CameraWorker cam A"]
    CM -- "start/stop" --> W2["CameraWorker cam B"]
    CAM["Camera RTSP"] --> FF["Tiến trình ffmpeg"]
    FF -- "raw BGR qua pipe" --> W1
    W1 --> DET["PersonDetector (YOLO + ByteTrack)"]
    DET --> AN["ProximityEvaluator (gom cụm)"]
    AN --> PUB["publisher.py"]
    PUB -- "bbox QoS0" --> VMS
    PUB -- "upload JPEG (HTTP)" --> ST["Storage Server"]
    PUB -- "crowd event QoS1" --> VMS
```

### 4.2. Mô hình luồng (thread / process)

| Luồng | Số lượng | Việc làm |
| :--- | :--- | :--- |
| Main thread | 1 | `client.loop_forever()`: xử lý mạng MQTT, callback |
| `camera-reconcile` | 1 mỗi lần nhận cấu hình | Parse payload, start/stop worker (tránh chặn thread MQTT) |
| `camera-<code>` | 1 mỗi camera | Vòng lặp AI: lấy frame, inference, gom cụm, publish |
| `rtsp-<code>` | 1 mỗi camera | Đọc raw frame từ stdout của ffmpeg |
| `rtsp-watch-<code>` | 1 mỗi camera | Watchdog: kill ffmpeg nếu stream treo |
| Tiến trình `ffmpeg` | 1 mỗi camera | Kết nối RTSP và giải mã video, chạy ngoài Python |
| `crowd-event-<code>` | tạm thời | Upload ảnh và publish event, không chặn vòng lặp AI |

**Model YOLO được chia sẻ** giữa các camera (cache trong `_SHARED_MODELS`), nhưng **mỗi camera có tracker riêng** (`_camera_trackers`). Số lần inference chạy song song bị giới hạn bởi `MAX_CONCURRENT_INFERENCES` (semaphore).

### 4.3. Luồng xử lý một frame

```mermaid
sequenceDiagram
    participant FF as ffmpeg process
    participant R as FFmpegFrameGrabber
    participant W as CameraWorker
    participant D as PersonDetector
    participant A as ProximityEvaluator
    participant P as publisher
    FF->>R: raw BGR frame (pipe)
    R->>R: lưu frame mới nhất, seq += 1
    W->>R: get_latest_frame(last_seq)
    W->>D: track(frame, latest_frame_getter)
    D->>R: peek_newer_frame() (lấy frame mới hơn nếu có)
    D-->>W: [(x1,y1,x2,y2,conf,track_id)]
    W->>A: clusters(dets)
    W->>P: publish_bbox (ngay lập tức)
    W->>W: CrowdPersistence.update()
    alt Đám đông đủ thời gian
        W->>P: publish_crowd_event (thread riêng)
    end
```

---

## 5. Chi tiết từng module

### 5.1. [`main.py`](main.py): entry point

- Gọi `setup_logging()`, in banner khởi động.
- `build_client()` tạo MQTT client, `connect()` rồi `loop_forever()`.
- Khi `Ctrl+C`: ngắt MQTT và gọi `stop_camera_workers()` để dừng mọi camera.

### 5.2. [`src/config.py`](src/config.py): cấu hình tập trung

- Đọc `.env` cho **thông tin kết nối** (bắt buộc: `MODULE`, `ZONE_ID`, `ORG_ID`, `MQTT_*`; tùy chọn: `STORAGE_URL`, `MQTT_TLS_ENABLED`).
- Mọi **tham số AI, đám đông, bbox, debug** được khai báo **trực tiếp trong file này** (chỉ có một nguồn sự thật, tránh sửa nhầm chỗ).
- Cuối file có phần **validate**: nếu giá trị sai (âm, bằng 0, sai enum) thì service dừng ngay lúc khởi động.

Chi tiết từng tham số ở [mục 7](#7-cấu-hình-và-điều-chỉnh-tham-số).

### 5.3. [`src/mqtt_client.py`](src/mqtt_client.py): kết nối và điều phối

- Transport **WebSockets**, path `MQTT_PATH`; bật TLS nếu `MQTT_TLS_ENABLED` (mặc định bật khi port 443).
- Tự reconnect (`reconnect_delay_set(1..30s)`).
- `on_connect`: subscribe `smart_vms/cameras/zone/{ZONE_ID}` (QoS 1).
- `on_message`: payload được đưa sang thread `camera-reconcile` để xử lý, **không làm chặn thread mạng MQTT**. `reconcile_lock` đảm bảo các lần reconcile chạy tuần tự.
- `on_start(camera)` / `on_stop(code)`: tạo/dừng `CameraWorker`, lưu trong dict `camera_workers`.
- `stop_camera_workers()`: gửi `signal_stop()` cho **tất cả** worker trước, rồi mới `join` từng cái (để tổng thời gian tắt ngắn hơn).

### 5.4. [`src/camera_manager.py`](src/camera_manager.py): reconcile danh sách camera

**Chọn camera:** trong payload `{"cameras": [...]}`, chỉ lấy camera có `ai_modules` chứa `MODULE` (vd. `CROWD_DETECTION`). URL RTSP lấy theo thứ tự ưu tiên:

1. `restream_urls[MODULE]` (luồng restream dành riêng cho module)
2. `rtsp_url` (luồng gốc)

Camera thiếu `id`, `code` hoặc URL sẽ bị bỏ qua (có log cảnh báo).

**Reconcile** (giống cách Kubernetes đưa trạng thái thực về trạng thái mong muốn):

| Trường hợp | Hành động |
| :--- | :--- |
| Có trong cấu hình mới, chưa chạy | Start |
| Đang chạy, không còn trong cấu hình | Stop |
| Đang chạy nhưng `id`/`rtsp_url` thay đổi | Stop rồi Start lại |
| Không đổi | Giữ nguyên |

### 5.5. [`src/camera_worker.py`](src/camera_worker.py): trái tim của hệ thống

#### a) `probe_stream()` và `FFmpegFrameGrabber`: đọc RTSP độ trễ thấp

**Nguyên lý:** dùng tiến trình `ffmpeg` riêng để giải mã, xuất **raw BGR24** ra `stdout`; Python đọc đúng `width*height*3` byte mỗi frame bằng `readinto()` vào mảng NumPy có sẵn (không copy thừa). Grabber **chỉ giữ frame mới nhất**; frame cũ bị ghi đè.

Lệnh ffmpeg (trong `_command()`):

```text
ffmpeg -rtsp_transport tcp -stimeout 4000000 -threads 2 -i <rtsp>
       -map 0:v:0 -an -sn -dn
       -vf scale=W:H:flags=fast_bilinear
       -vsync 0 -c:v rawvideo -pix_fmt bgr24 -threads:v 1 -f rawvideo pipe:1
```

| Tùy chọn | Ý nghĩa |
| :--- | :--- |
| `-rtsp_transport tcp` | Không mất gói như UDP (tránh lỗi HEVC `Could not find ref with POC`) |
| `-stimeout 4000000` | Timeout socket 4 giây |
| `-threads 2` | Ít thread giải mã: hàng đợi frame nông hơn, không tranh CPU với YOLO |
| `-map 0:v:0 -an -sn -dn` | Chỉ lấy video, bỏ audio/sub/data |
| `-vsync 0` | Đẩy frame ra ngay, không nhân/bỏ frame để căn timestamp |
| `-pix_fmt bgr24` | Đúng định dạng OpenCV/YOLO dùng, không cần chuyển màu trong Python |

`W:H` lấy từ `ffprobe` (`_ensure_dimensions()`), tức kích thước gốc của stream.

**Cơ chế an toàn:**

| Cơ chế | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `max_frame_age` | 0.40s | Frame cũ hơn ngưỡng này sẽ **không** được đưa vào AI |
| `startup_timeout` | 10s | Watchdog kill ffmpeg nếu không ra frame đầu tiên |
| `stall_timeout` | 3.5s | Watchdog kill ffmpeg nếu đang chạy mà đứng hình |
| `reconnect_delay` | 0.5s | Thời gian chờ trước khi mở lại ffmpeg |
| Đếm lỗi stderr | | `reference_errors` (lỗi tham chiếu HEVC), `decoder_errors` |

API cho vòng lặp AI:

- `get_latest_frame(last_seq, timeout)`: chờ đến khi có frame **mới hơn** `last_seq` và **còn tươi**.
- `peek_newer_frame(current_seq)`: không chờ; trả frame mới hơn nếu có (dùng ngay trước inference để lấy frame mới nhất).

> [!IMPORTANT]
> **Vì sao không dùng `cv2.VideoCapture`?** Khi YOLO chiếm CPU/GIL, thread gọi `capture.read()` không kịp đọc, gói tin dồn lại trong buffer của libavformat/TCP. `CAP_PROP_BUFFERSIZE=1` không xóa được buffer này, nên frame đọc được **cũ hơn thực tế 0.5 đến 2 giây**. Triệu chứng: bbox trên VMS di chuyển **đúng tốc độ** của người nhưng **luôn đi sau**. Tách ffmpeg ra tiến trình riêng giải quyết triệt để vấn đề này.

#### b) `CameraWorker._run()`: vòng lặp AI

Thứ tự xử lý (được sắp xếp để **bbox lên VMS sớm nhất có thể**):

1. `cv2.setNumThreads(1)`: tránh OpenCV bung nhiều thread tranh CPU.
2. Lấy frame mới nhất từ grabber.
3. `detector.track(frame, latest_frame_getter=...)`: ngay trước inference, đổi sang frame mới hơn nếu có.
4. `analyzer.clusters(dets)` → lọc cụm có `len >= NUMBER_PERSON_IS_CROWDED`.
5. `_build_bbox_detections()` → **`publish_bbox()` ngay lập tức**.
6. Sau đó mới làm phần chậm hơn: `CrowdPersistence.update()`, vẽ ảnh, encode JPEG, gửi event, ghi debug.
7. Cứ 5 giây log hiệu năng một lần (`Pipeline performance ...`).

#### c) `_build_bbox_detections()`: quy tắc chọn khung gửi lên VMS

- Tọa độ được **chuẩn hóa về 0..1** theo kích thước frame.
- **Không có đám đông:** gửi tất cả người (khung xanh `#00ff00`) nếu `PUBLISH_ALL_PERSON_BBOX=True`, ngược lại không gửi khung nào.
- **Có đám đông:** chỉ gửi người **thuộc cụm đông** (khung đỏ `#ff0000`) + **khung bao cụm** `crowd_N` (nới rộng thêm `CROWD_BBOX_PADDING_PERCENT`%).

#### d) `CrowdPersistence`: chống báo động giả

Một cụm đông chỉ được báo khi **tồn tại liên tục ≥ `CROWD_PERSISTENCE_SECONDS`**. Thuật toán:

1. Mỗi frame, mỗi cụm đông được **so khớp** với các "ứng viên" (`_CrowdCandidate`) đang theo dõi bằng điểm số:
   - `10 x` số `track_id` trùng hiện tại
   - `3 x` số `track_id` trùng trong lịch sử
   - `5 x IoU` khung cụm
   - `+ (1 - khoảng cách tâm / kích thước)`
   
   Điều kiện khớp: có `track_id` trùng **hoặc** `IoU >= CROWD_SPATIAL_IOU_THRESHOLD` **hoặc** tỉ lệ khoảng cách tâm `<= CROWD_CENTER_DISTANCE_RATIO`. Nhờ vậy, khi ByteTrack đổi ID hoặc người ra vào cụm, ứng viên vẫn không bị reset.
2. Không khớp → tạo ứng viên mới (`started_at = now`).
3. Ứng viên **mất dấu quá `CROWD_TRACK_GRACE_SECONDS`** (hoặc 2.5 x khoảng cách frame nếu FPS thấp) → xóa.
4. Khi `now - started_at >= CROWD_PERSISTENCE_SECONDS` → đưa vào danh sách `ready` để gửi event.
5. Sau lần gửi thành công đầu tiên, **lặp lại** mỗi `CROWD_EVENT_REPEAT_INTERVAL_SECONDS` nếu đám đông vẫn còn. Gửi lỗi thì **thử lại sau 5 giây** (`CROWD_EVENT_RETRY_DELAY_S`).

```mermaid
stateDiagram-v2
    [*] --> Candidate: phát hiện cụm >= N người
    Candidate --> Candidate: vẫn thấy (khớp ID/IoU/tâm)
    Candidate --> [*]: mất dấu > grace
    Candidate --> Publishing: tồn tại >= persistence
    Publishing --> Notified: gửi OK
    Publishing --> Candidate: gửi lỗi (thử lại sau 5s)
    Notified --> Publishing: còn đông sau repeat_interval
    Notified --> [*]: mất dấu > grace
```

Việc gửi event chạy trong **thread riêng** (`_publish_crowd_event_async`); kết quả trả về qua `queue.Queue` và được xử lý ở frame kế tiếp (`complete_publish`).

### 5.6. [`src/detector.py`](src/detector.py): `PersonDetector`

#### Chọn backend (`_select_runtime`)

```mermaid
flowchart TD
    S["YOLO_INFERENCE_BACKEND"] --> C{"Có CUDA?"}
    C -- "Có" --> T{"backend = tensorrt và có file .engine?"}
    T -- "Có" --> TRT["TensorRT"]
    T -- "Không" --> PTG["PyTorch GPU (half nếu bật)"]
    C -- "Không" --> O{"backend auto/openvino và có thư mục OpenVINO?"}
    O -- "Có" --> OV["OpenVINO (imgsz theo metadata nếu model static)"]
    O -- "Không" --> PTC["PyTorch CPU"]
```

Thư mục OpenVINO được tìm theo thứ tự: `OPENVINO_MODEL_PATH` → `<tên weight>_openvino_model` cạnh file `.pt` → chính `weights_path` (nếu là thư mục).

> [!WARNING]
> Hiện chỉ có `best_openvino_model`, **chưa có `final_openvino_model`**. Trên máy không có GPU, `final.pt` sẽ chạy bằng **PyTorch CPU**. Muốn dùng OpenVINO thì export: `yolo export model=resources/runs/train/weights/final.pt format=openvino imgsz=640`. Nếu model OpenVINO là static, `imgsz` sẽ **lấy từ metadata**, không lấy từ `YOLO_IMGSZ`.

#### Model dùng chung và tracker riêng

- `_SHARED_MODELS`: cache theo `(backend, device, path, imgsz, half)`; nhiều camera dùng chung 1 model, tiết kiệm RAM/VRAM.
- Ultralytics lưu tracker trong `predictor.trackers` (đối tượng dùng chung). `track()` **hoán đổi** tracker của camera hiện tại vào trước khi gọi `model.track(persist=True)` và lưu lại sau đó, nên ID không bị lẫn giữa các camera. Việc này nằm trong `_INFERENCE_SEMAPHORE` để an toàn khi chạy nhiều thread.

#### Hàm chính

| Hàm | Đầu ra | Dùng khi |
| :--- | :--- | :--- |
| `predict(frame)` | `[(x1,y1,x2,y2,conf)]` | Chỉ detect, không cần ID |
| `track(frame, latest_frame_getter)` | `[(x1,y1,x2,y2,conf,track_id)]` | Vòng lặp chính (YOLO + ByteTrack) |
| `visualize(frame, dets)` | ảnh đã vẽ | Debug |

Tọa độ trả về theo **pixel của frame gốc**. Chỉ lọc class có tên `person` (không phân biệt hoa thường); model không có class này thì báo lỗi ngay khi khởi tạo.

### 5.7. [`src/analyze.py`](src/analyze.py): `ProximityEvaluator` (gom cụm)

**Nguyên lý:** hai người được xem là "đứng gần nhau" nếu **khoảng cách giữa hai điểm chân** nhỏ hơn **α lần chiều dài cơ thể trung bình** của hai người đó. Dùng chiều dài cơ thể làm thước đo giúp ngưỡng **tự co giãn theo khoảng cách tới camera** (người ở xa nhỏ hơn thì ngưỡng cũng nhỏ hơn).

- Điểm chân: `foot = ((x1 + x2) / 2, y2)`
- Chiều dài cơ thể: `L = max(w, h)` (mode `max`) hoặc `L = h` (mode `height`)
- Khoảng cách (bù phối cảnh trục y):

  `d(i,j) = sqrt( (fx_i - fx_j)^2 + (y_scale * (fy_i - fy_j))^2 )`
- Ngưỡng: `thr(i,j) = alpha * (L_i + L_j) / 2`
- `i` và `j` lân cận ⇔ `d(i,j) <= thr(i,j)`

`y_scale > 1` vì camera nhìn chéo xuống, khoảng cách theo chiều sâu bị **nén** trên trục y của ảnh.

**Gom cụm:** dựng ma trận kề `N x N` (tính vector hóa bằng NumPy), sau đó tìm **thành phần liên thông** bằng DFS. Cụm có tính bắc cầu: A gần B, B gần C thì A, B, C cùng một cụm dù A và C cách xa nhau.

### 5.8. [`src/publisher.py`](src/publisher.py): gửi dữ liệu

- `publish_bbox(...)`: QoS 0, bỏ qua nếu MQTT chưa kết nối; dùng `_MQTT_PUBLISH_LOCK` để nhiều camera publish an toàn. `camera_code` được thay `/ # +` bằng `_` để làm topic hợp lệ.
- `upload_snapshot(jpg, camera_id)`: `POST multipart` tới `STORAGE_URL` (timeout 10s), trả về `object_key`.
- `publish_crowd_event(...)`: **bắt buộc có ảnh và upload thành công**, sau đó publish QoS 1 và chờ broker ACK tối đa 5s. Bất kỳ bước nào lỗi đều trả về `False` để `CrowdPersistence` thử lại.

### 5.9. [`src/setup_logging.py`](src/setup_logging.py)

- Ghi ra `logs/app.log` (xoay vòng 5MB x 3 bản) và console, level lấy từ `LOG_LEVEL`.
- Filter `MaskCredentials` thay `rtsp://user:pass@` thành `rtsp://***:***@` trên **mọi** dòng log.

---

## 6. Giao tiếp MQTT với VMS

### 6.1. Bảng topic

| Hướng | Topic | QoS | Nội dung |
| :--- | :--- | :--- | :--- |
| VMS → AI | `smart_vms/cameras/zone/{ZONE_ID}` | 1 | Danh sách camera của zone |
| AI → VMS | `smart_vms/ai/bbox/{camera_code}` | 0 | BBox realtime |
| AI → VMS | `smart_vms/ai_events/{MODULE}` | 1 | Sự kiện đám đông |

### 6.2. Payload nhận: danh sách camera

```json
{
  "cameras": [
    {
      "id": "123",
      "code": "nhatron-loira",
      "ai_modules": ["CROWD_DETECTION"],
      "rtsp_url": "rtsp://user:pass@host/stream",
      "restream_urls": { "CROWD_DETECTION": "rtsp://restream-host/..." }
    }
  ]
}
```

### 6.3. Payload gửi: BBox

```json
{
  "camera_code": "nhatron-loira",
  "camera_id": "123",
  "organization_id": 115,
  "ai_modules": "CROWD_DETECTION",
  "timestamp": 1762740000.123,
  "detections": [
    {
      "id": "person_7", "label": "person_7 0.83", "class": "person",
      "confidence": 0.83, "bbox": [0.41, 0.22, 0.47, 0.61],
      "color": "#ff0000", "cluster_id": 1, "track_id": 7
    },
    {
      "id": "crowd_1", "label": "crowd_1 (5 people)", "class": "crowd",
      "bbox": [0.30, 0.15, 0.62, 0.70], "color": "#ff0000",
      "cluster_id": 1, "person_count": 5, "track_ids": [3, 5, 7, 9, 11]
    }
  ],
  "hud": { "text": "CROWD: 1 cluster(s)", "display_ms": 1000, "color": "#ff0000" }
}
```

`bbox = [x1, y1, x2, y2]` chuẩn hóa 0..1. Khi không có người, `detections` là `[]` (VMS sẽ xóa khung đang hiển thị).

### 6.4. Payload gửi: Crowd event

```json
{
  "ai_modules": "CROWD_DETECTION",
  "submodule_code": "CROWD_DETECTION",
  "camera_id": "123",
  "camera_code": "nhatron-loira",
  "event_time": "2026-11-11T10:30:00+07:00",
  "event_type": "CROWD_DETECTION",
  "entity_type": "PERSON",
  "severity": "HIGH",
  "payload": {
    "type": "HIGH_DENSITY",
    "person_count": 5,
    "threshold": 5,
    "clusters": [
      { "cluster_id": 1, "person_count": 5, "detections": [0,1,2,3,4], "track_ids": [3,5,7,9,11] }
    ]
  },
  "snapshot_url": "<object_key trả về từ Storage>"
}
```

---

## 7. Cấu hình và điều chỉnh tham số

> [!IMPORTANT]
> **Quy tắc:** `.env` chỉ chứa **thông tin kết nối/bí mật**. Mọi tham số thuật toán sửa trong [`src/config.py`](src/config.py). Không khai báo trùng ở hai nơi. Sửa xong phải **khởi động lại service**.

### 7.1. `.env`: kết nối

| Biến | Bắt buộc | Ví dụ | Ý nghĩa |
| :--- | :---: | :--- | :--- |
| `ZONE_ID` | ✔ | `289` | Zone cần lắng nghe danh sách camera |
| `ORG_ID` | ✔ | `115` | Mã tổ chức (số nguyên), gửi kèm bbox |
| `MODULE` | ✔ | `CROWD_DETECTION` | Tên module AI, dùng để lọc camera và đặt tên topic event |
| `MQTT_HOST` / `MQTT_PORT` / `MQTT_PATH` | ✔ | `192.168.1.250` / `19048` / `/mqtt` | Broker MQTT over WebSocket |
| `MQTT_USER` / `MQTT_PASS` | ✔ | | Tài khoản broker |
| `MQTT_CLIENT_ID` | ✔ | | **Phải duy nhất**; hai instance trùng ID sẽ đá nhau liên tục |
| `MQTT_TLS_ENABLED` | | `true/false` | Mặc định `true` nếu port 443 |
| `STORAGE_URL` | | `https://.../storage/upload/file` | API upload ảnh; **phải truy cập được từ máy chạy AI** |

### 7.2. Model và inference

| Tham số | Hiện tại | Ảnh hưởng khi tăng / giảm |
| :--- | :--- | :--- |
| `WEIGHT_YOLO_MODEL_PATH` | `./resources/runs/train/weights/final.pt` | Đổi model |
| `OPENVINO_MODEL_PATH` | `.../final_openvino_model` | Đường dẫn model OpenVINO (CPU) |
| `YOLO_INFERENCE_BACKEND` | `auto` | `auto`/`pytorch`/`cuda`/`openvino`/`tensorrt` |
| `YOLO_OPENVINO_DEVICE` | `intel:cpu` | `intel:gpu`, `auto` (tự fallback về CPU nếu không có GPU) |
| `YOLO_IMGSZ` | `640` | **Nên khớp với imgsz lúc train.** Tăng: bắt người xa/nhỏ tốt hơn nhưng chậm hơn (gần như x4 khi lên 1280) |
| `YOLO_MODEL_CONF` | `0.6` | Tăng: ít báo nhầm nhưng dễ sót người xa/che khuất. Giảm: bắt nhiều hơn nhưng dễ nhận nhầm vật thể |
| `YOLO_HALF_PRECISION` | `True` | FP16, **chỉ có tác dụng trên CUDA** |
| `YOLO_CPU_THREADS` | `4` | Số thread PyTorch trên CPU; nên ≈ số nhân vật lý trừ đi phần cho ffmpeg |
| `MAX_CONCURRENT_INFERENCES` | `1` | Số inference chạy song song (nhiều camera). Tăng chỉ có lợi khi GPU/CPU còn dư |

### 7.3. Gom cụm và đám đông

| Tham số | Hiện tại | Ý nghĩa / Hướng chỉnh |
| :--- | :--- | :--- |
| `PROXIMITY_ALPHA` | `1.5` | Ngưỡng gần = α x chiều dài cơ thể. **Tăng** thì dễ gom cụm hơn (nhạy); **giảm** thì người phải đứng sát mới tính |
| `PROXIMITY_Y_SCALE` | `1.5` | Bù phối cảnh trục y. Camera càng nghiêng thì đặt càng lớn; camera nhìn thẳng từ trên xuống đặt ≈ `1.0` |
| `PROXIMITY_LENGTH_MODE` | `max` | `max(w,h)` ổn định khi người ngồi/cúi; `height` chặt hơn khi người đứng |
| `NUMBER_PERSON_IS_CROWDED` | `5` | Số người tối thiểu để một cụm bị coi là đám đông |
| `CROWD_PERSISTENCE_SECONDS` | `4.0` | Đám đông phải tồn tại bao lâu mới gửi event. Tăng thì ít báo giả nhưng báo chậm hơn |
| `CROWD_TRACK_GRACE_SECONDS` | `2.5` | Thời gian chịu mất dấu (do che khuất, nhấp nháy) trước khi reset bộ đếm |
| `CROWD_SPATIAL_IOU_THRESHOLD` | `0.25` | IoU tối thiểu để coi là cùng một đám đông giữa các frame |
| `CROWD_CENTER_DISTANCE_RATIO` | `0.6` | Khoảng cách tâm / kích thước cụm tối đa để coi là cùng một đám đông |
| `CROWD_EVENT_REPEAT_INTERVAL_SECONDS` | `10.0` | Chu kỳ gửi lại event khi đám đông vẫn còn |
| `CROWD_BBOX_PADDING_PERCENT` | `10.0` | Nới rộng khung `crowd_N` thêm bao nhiêu % |

### 7.4. BBox, log, debug

| Tham số | Hiện tại | Ý nghĩa |
| :--- | :--- | :--- |
| `PUBLISH_ALL_PERSON_BBOX` | `True` | Gửi khung của mọi người khi chưa có đám đông |
| `LOG_LEVEL` | `INFO` | `DEBUG` để xem cả lỗi publish bbox |
| `DEBUG_VIDEO_ENABLED` | `False` | Ghi video đã vẽ khung vào `resources/debug_videos/` (**tốn CPU**) |
| `DEBUG_FRAME_SAVE_ENABLED` | `False` | Lưu ảnh mỗi 5 giây vào `resources/img_test/` |
| `DEBUG_VIDEO_FPS` | `10.0` | FPS của video debug |
| `DEBUG_MAX_SAVED_FRAMES` / `DEBUG_MAX_SAVED_VIDEOS` | `200` / `10` | Tự xóa file cũ khi vượt số lượng |

### 7.5. Tham số trong code (ít khi cần sửa)

| Vị trí | Tham số | Mặc định |
| :--- | :--- | :--- |
| `FFmpegFrameGrabber.__init__` | `max_frame_age`, `stall_timeout`, `startup_timeout`, `reconnect_delay`, `decoder` | 0.40s, 3.5s, 10s, 0.5s, `None` |
| `FFmpegFrameGrabber._command` | `-threads 2`, `scale=...` | |
| `camera_worker.py` | `CROWD_EVENT_RETRY_DELAY_S`, `PERFORMANCE_LOG_INTERVAL_S`, `DEBUG_FRAME_SAVE_INTERVAL_S` | 5s, 5s, 5s |
| `publisher.py` | timeout upload, timeout chờ ACK, `hud.display_ms` | 10s, 5s, 1000ms |
| `detector.py` | Tracker | `bytetrack.yaml` (mặc định của Ultralytics) |

> [!TIP]
> Có GPU NVIDIA và muốn giảm CPU dành cho giải mã: truyền `decoder="hevc_cuvid"` (hoặc `h264_cuvid`) vào `FFmpegFrameGrabber(...)` trong `CameraWorker._run()`. Với ảnh 2K (2688x1520), có thể giảm tải pipe/RAM bằng cách truyền `width`/`height` nhỏ hơn (vd. `1344x760`), vì YOLO đằng nào cũng resize về `640`.

### 7.6. Gợi ý tinh chỉnh theo triệu chứng

| Triệu chứng | Thử chỉnh |
| :--- | :--- |
| Sót người ở xa/nhỏ | Giảm `YOLO_MODEL_CONF` (0.5 → 0.45), hoặc train/export model ở `imgsz` lớn hơn |
| Nhận nhầm vật thể là người | Tăng `YOLO_MODEL_CONF`, bổ sung ảnh âm tính vào dataset |
| Người đứng rải rác vẫn bị gom cụm | Giảm `PROXIMITY_ALPHA` hoặc `PROXIMITY_Y_SCALE` |
| Nhóm đứng sát nhau mà không thành cụm | Tăng `PROXIMITY_ALPHA` |
| Báo động giả thoáng qua | Tăng `CROWD_PERSISTENCE_SECONDS` |
| Đám đông thật mà bộ đếm cứ bị reset | Tăng `CROWD_TRACK_GRACE_SECONDS`, `CROWD_CENTER_DISTANCE_RATIO` |
| FPS thấp, bbox giật | Dùng GPU/OpenVINO, giảm `YOLO_IMGSZ`, tắt debug video |

---

## 8. Cài đặt và chạy

### 8.1. Cài đặt

```bash
# 1. Cài ffmpeg (bắt buộc)
#    Ubuntu: sudo apt install ffmpeg
#    Windows: tải bản build ffmpeg và thêm thư mục bin vào PATH
ffmpeg -version && ffprobe -version

# 2. Tạo môi trường Python 3.11
python -m venv .venv
# Linux: source .venv/bin/activate   |   Windows: .\.venv\Scripts\activate

# 3. Cài thư viện
pip install ultralytics opencv-python numpy paho-mqtt requests python-dotenv pyyaml
# Tùy chọn: pip install openvino   (CPU Intel)
# GPU: cài torch bản CUDA theo hướng dẫn trên pytorch.org
```

### 8.2. Cấu hình

1. Tạo `.env` ở thư mục gốc (xem [7.1](#71-env-kết-nối)).
2. Kiểm tra file model tại `WEIGHT_YOLO_MODEL_PATH`.
3. Kiểm tra tham số trong `src/config.py`.

### 8.3. Chạy

```bash
python main.py
```

Log khởi động bình thường sẽ có dạng:

```text
Connected: Success
Send Subscribe sucessfull!, topics=smart_vms/cameras/zone/289
Camera bật module CROWD_DETECTION: ['nhatron-loira']
Loaded shared YOLO model backend=pytorch device=cpu model=... imgsz=640
Opened RTSP stream info camera=nhatron-loira size=2688x1520 codec=hevc fps=20/1
Pipeline performance camera=nhatron-loira frames=.. fps=.. stage_avg/max_ms=[read=.. infer=.. ...]
```

Không cần cấu hình camera bằng tay: **thêm/bớt camera hoặc bật/tắt module trên VMS**, service sẽ tự start/stop worker tương ứng.

---

## 9. Logging, theo dõi hiệu năng, debug

**Log hiệu năng** (mỗi 5 giây cho mỗi camera):

```text
Pipeline performance camera=X frames=42 fps=8.40 cpu_threads=4 bbox_publish_ok=42/42
stage_avg/max_ms=[read=3/20 infer=95/140 cluster=0/1 bbox_build=0/1 bbox_publish=0/2 crowd=0/1 ...]
```

| Stage | Ý nghĩa | Bất thường khi |
| :--- | :--- | :--- |
| `read` | Thời gian chờ frame mới | Cao: camera/mạng chậm, hoặc FPS camera thấp |
| `infer` | YOLO + ByteTrack | Cao: cần GPU/OpenVINO hoặc giảm imgsz |
| `bbox_publish` | Đưa bbox vào hàng đợi MQTT | Cao: mạng tới broker có vấn đề |
| `loop` | Tổng một vòng | `1000 / loop` ≈ FPS xử lý |
| `bbox_publish_ok` | Số lần publish thành công / tổng | Nhỏ hơn tổng: MQTT bị mất kết nối |

**Log nghiệp vụ quan trọng:** `Cluster summary` (mỗi giây), `Crowd candidate detected`, `Persistent crowd detected`, `Snapshot uploaded`, `MQTT crowd event broker-acknowledged`, `Crowd cleared`, `FFmpeg watchdog restarting stalled stream`.

**Debug trực quan:** bật `DEBUG_FRAME_SAVE_ENABLED` hoặc `DEBUG_VIDEO_ENABLED` để xem khung, cụm, thời gian inference và FPS được vẽ lên ảnh.

---

## 10. Các lỗi thường gặp

| Hiện tượng / Log | Nguyên nhân | Cách xử lý |
| :--- | :--- | :--- |
| `Cannot probe RTSP camera=...` | Sai URL/mật khẩu, camera offline, thiếu `ffprobe` | Thử `ffprobe -rtsp_transport tcp <url>` trên máy chạy AI |
| `FFmpeg watchdog restarting stalled stream` | Stream đứng hình | Kiểm tra mạng/camera; service sẽ tự kết nối lại |
| `Upload ảnh lỗi ... Connection refused` | `STORAGE_URL` không truy cập được từ máy AI (sai host/port, DNS trỏ sai, firewall) | Sửa `STORAGE_URL` trong `.env`; test bằng `curl -I <STORAGE_URL>` |
| `Crowd event not published because snapshot upload failed` | Hệ quả của lỗi upload (event bắt buộc phải có ảnh) | Như trên; event sẽ tự thử lại mỗi 5 giây |
| `MQTT crowd event timeout` | Broker không ACK trong 5 giây | Kiểm tra broker/mạng |
| MQTT liên tục `Disconnected` rồi `Connected` | Hai instance trùng `MQTT_CLIENT_ID` | Đặt ID duy nhất cho mỗi instance |
| `Model classes not contain 'person'` | Model không có class `person` | Dùng đúng model đã train |
| `Model path not found` | Sai `WEIGHT_YOLO_MODEL_PATH` | Đường dẫn tương đối tính từ **thư mục chạy lệnh**; hãy chạy `python main.py` từ thư mục gốc |
| Bbox trên VMS bị trễ so với hình | Frame bị dồn ứ (CPU quá tải) | Xem `infer` trong log hiệu năng; giảm tải hoặc dùng GPU |

---

## 11. Bài học từ quá trình tối ưu

Đây là những kết luận đã được kiểm chứng trong quá trình phát triển. Nên đọc trước khi sửa phần đọc video hoặc tracking:

1. **Nguyên nhân chính của bbox trễ là khâu đọc RTSP, không phải AI hay MQTT.** `cv2.VideoCapture` để dồn frame cũ khi CPU bận; chuyển sang tiến trình `ffmpeg` riêng + chỉ giữ frame mới nhất + kiểm tra `max_frame_age` đã giải quyết vấn đề.
2. **Không cần (và không nên) dự đoán bù chuyển động** bằng vận tốc Kalman, cũng không cần thread heartbeat gửi lại bbox cũ. Những cách này từng được thử, nhưng không cải thiện mà còn làm khung lệch/kéo lùi. Hiện tại: **mỗi frame inference → gửi bbox đúng một lần**.
3. **Không thêm `fflags nobuffer` / `max_delay 0`** cho RTSP: các cờ này đã làm thời gian mở stream lên tới ~11 giây và FPS rơi xuống ~1.4.
4. **`imgsz` phải khớp lúc train.** `final.pt` được train với `imgsz=640`. Đổi `imgsz` khi suy luận làm phân bố confidence thay đổi (người ở xa có thể tụt dưới ngưỡng `conf`).
5. **Ngưỡng `conf` và ByteTrack ảnh hưởng nhau:** khi truyền `conf` cao vào `model.track()`, detection có confidence dưới ngưỡng bị loại **trước khi** vào ByteTrack. Track mới cũng cần được xác nhận qua vài frame mới được trả về. Vì vậy, nếu thấy `track()` trả về ít người hơn `predict()`, hãy thử giảm `YOLO_MODEL_CONF` trước.
6. **Mọi tham số chỉ sửa ở một nơi** (`src/config.py`), để tránh trường hợp sửa xong nhưng không có tác dụng.
