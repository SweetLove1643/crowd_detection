import cv2
from pathlib import Path

# ====== Cấu hình ======
SAVE_EVERY_SEC = 0.8      # lưu 1 frame mỗi 0.5 giây (đặt None nếu muốn dùng STEP_FRAMES)
STEP_FRAMES = 15          # dùng khi SAVE_EVERY_SEC = None
JPG_QUALITY = 100          # chất lượng ảnh JPEG (0-100)

# Thư mục output
dir_output = Path(__file__).parent.parent / "resources" / "frames"
dir_output.mkdir(parents=True, exist_ok=True)

# Video input
video_path = Path(__file__).parent.parent / "resources" / "recordings" / "nhatron-loira_20261105_163141.mp4"

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Tính bước lưu theo số frame
if SAVE_EVERY_SEC is not None and fps > 0:
    step = max(1, round(fps * SAVE_EVERY_SEC))
else:
    step = STEP_FRAMES

print(f"Video FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Saving 1 frame every {step} frames (~{total_frames // step} images)")

# Tiền tố tên file = tên video, tránh trùng khi gom nhiều video
prefix = video_path.stem

frame_idx = 0
saved = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Finished reading video.")
            break

        if frame_idx % step == 0:
            out_path = dir_output / f"{prefix}_{frame_idx:06d}.jpg"
            cv2.imwrite(
                str(out_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY],
            )
            saved += 1

        frame_idx += 1

finally:
    cap.release()

print(f"Done. Saved {saved} frames to: {dir_output}")