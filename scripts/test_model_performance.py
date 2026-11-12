"""Test YOLO26n trên video: vẽ bbox + confidence.

Cài đặt:  pip install ultralytics opencv-python
Chạy:     python ./scripts/test_model_performance.py --source ./resources/recordings/sanquangtruong_test.mkv
          python ./scripts/test_model_performance.py --source ./resources/recordings/test_model.mp4 --weights ./resources/runs/train/weights/final2.pt --conf 0.55 --save ./resources/output/out2.mp4
Phím:     q hoặc ESC để thoát
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Đường dẫn video (hoặc 0 cho webcam)")
    p.add_argument("--weights", default="yolo26n.pt", help="File weights (mặc định: yolo26n.pt)")
    p.add_argument("--conf", type=float, default=0.55, help="Ngưỡng confidence")
    p.add_argument("--imgsz", type=int, default=896)
    p.add_argument("--iou", type=int, default=0.7)
    p.add_argument("--device", default=None, help="vd: 0 hoặc cpu (mặc định: tự chọn)")
    p.add_argument("--save", default=None, help="Đường dẫn lưu video kết quả (tuỳ chọn)")
    p.add_argument("--no-show", action="store_true", help="Không hiển thị cửa sổ")
    return p.parse_args()


def color_for(cls_id: int):
    # Màu cố định theo class
    return (37 * cls_id % 256, 17 * cls_id % 256 + 100, 91 * cls_id % 256)


def main():
    args = parse_args()
    model = YOLO(args.weights)
    names = model.names

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        result = model.predict(
            frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
        )[0]
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            color = color_for(cls_id)
            label = f"{names[cls_id]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            y_top = max(y1 - th - bl - 4, 0)
            cv2.rectangle(frame, (x1, y_top), (x1 + tw + 4, y_top + th + bl + 4), color, -1)
            cv2.putText(frame, label, (x1 + 2, y_top + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("YOLO26n", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()