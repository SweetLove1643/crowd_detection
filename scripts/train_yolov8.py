"""
Pipeline: CVAT (LabelMe 3.0) -> YOLO format -> letterbox 1280 -> split train/val
          -> train YOLOv8 (CPU) -> export OpenVINO -> so sánh PyTorch vs OpenVINO

Cài đặt:
    pip install ultralytics openvino pyyaml opencv-python
    (chỉ khi EXPORT_INT8 = True:  pip install nncf)

Cấu trúc thư mục (cạnh thư mục scripts/):
    resources/cvat_export/   <- giải nén file export của CVAT vào đây (ảnh + xml/json)
    resources/dataset_yolo/  <- script tự tạo
    resources/runs/          <- kết quả train + model export
    resources/debug/         <- ảnh/text để kiểm tra
"""

import json
import random
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

# ============================ CONFIG ============================
ROOT = Path(__file__).parent.parent / "resources"
RAW_DIR = ROOT / "cvat_export"
DATASET_DIR = ROOT / "dataset_yolo"
RUNS_DIR = ROOT / "runs"
DEBUG_DIR = ROOT / "debug"

TASK = "detect"            # "detect" (bbox) hoặc "segment" (polygon)
CLASS_NAMES = None         # None = tự lấy từ annotation (sắp xếp A-Z). Hoặc ["scratch", "dirty", ...]

IMG_SIZE = 1280
PRE_LETTERBOX = True       # True: letterbox ảnh về IMG_SIZE x IMG_SIZE (pad xám) rồi lưu vào dataset
                           # False: giữ ảnh gốc, để Ultralytics tự letterbox khi train
PAD_COLOR = (114, 114, 114)

VAL_RATIO = 0.2
SEED = 42

MODEL_WEIGHTS = "yolov8n.pt" if TASK == "detect" else "yolov8n-seg.pt"
EPOCHS = 100
BATCH = 4
WORKERS = 2
PATIENCE = 30
CACHE = "ram"              # "ram" | "disk" | False
RUN_NAME = "train"

EXPORT_INT8 = False        # INT8 nhanh hơn nhưng cần nncf, dataset nhỏ có thể kém ổn định -> thử sau
PREDICT_CONF = 0.25

STEP_PREPARE = True        # convert + split
STEP_TRAIN = True
STEP_EXPORT = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# ================================================================


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def color_of(cid):
    hsv = np.uint8([[[(cid * 47) % 180, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


# ---------------------------------------------------------------
# 1. Đọc annotation (CVAT "LabelMe 3.0" = XML; hỗ trợ thêm JSON của labelme)
# ---------------------------------------------------------------
def parse_xml(path):
    root = ET.parse(path).getroot()
    dw = int(float(root.findtext("imagesize/ncols") or 0))
    dh = int(float(root.findtext("imagesize/nrows") or 0))
    objects = []
    for obj in root.findall("object"):
        if (obj.findtext("deleted") or "0").strip().lower() in ("1", "true", "yes"):
            continue
        name = (obj.findtext("name") or "").strip()
        pts = [(float(p.findtext("x")), float(p.findtext("y"))) for p in obj.findall("polygon/pt")]
        if not pts:
            box = obj.find("segm/box")
            if box is not None:
                x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        objects.append((name, pts))
    return dw, dh, objects


def parse_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    dw = int(data.get("imageWidth") or 0)
    dh = int(data.get("imageHeight") or 0)
    objects = []
    for s in data.get("shapes", []):
        st = s.get("shape_type", "polygon")
        pts = [(float(x), float(y)) for x, y in s["points"]]
        if st == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        elif st not in ("polygon", "rectangle"):
            print(f"  [WARN] {path.name}: bỏ shape_type '{st}' (chưa hỗ trợ)")
            continue
        objects.append((s["label"].strip(), pts))
    return dw, dh, objects


def load_samples():
    section("[DEBUG 1] Quét dữ liệu CVAT")
    imgs = {p.stem: p for p in RAW_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS}
    anns = {p.stem: p for p in RAW_DIR.rglob("*") if p.suffix.lower() in (".xml", ".json")}
    print(f"Thư mục: {RAW_DIR}")
    print(f"Tìm thấy {len(imgs)} ảnh, {len(anns)} file annotation")

    samples = []
    for stem, ap in sorted(anns.items()):
        if stem not in imgs:
            print(f"  [WARN] {ap.name}: không có ảnh tương ứng -> bỏ qua")
            continue
        dw, dh, objs = parse_xml(ap) if ap.suffix.lower() == ".xml" else parse_json(ap)
        clean = []
        for label, pts in objs:
            if not label or len(pts) < 3:
                print(f"  [WARN] {ap.name}: object lỗi (label='{label}', {len(pts)} điểm) -> bỏ")
                continue
            clean.append((label, pts))
        samples.append(dict(stem=stem, img=imgs[stem], dw=dw, dh=dh, objects=clean))
    for stem in sorted(set(imgs) - set(anns)):
        print(f"  [WARN] {imgs[stem].name}: không có annotation -> bỏ qua")

    if not samples:
        raise RuntimeError("Không có cặp ảnh/annotation nào. Kiểm tra RAW_DIR.")

    print(f"\n{'stem':45s} {'#obj':>5s}  labels")
    for s in samples:
        cnt = Counter(l for l, _ in s["objects"])
        print(f"{s['stem']:45s} {len(s['objects']):5d}  {dict(cnt)}")
    total = Counter(l for s in samples for l, _ in s["objects"])
    print(f"\nTổng {len(samples)} ảnh | instances theo class: {dict(total)}")
    empty = [s["stem"] for s in samples if not s["objects"]]
    if empty:
        print(f"Ảnh không có object (dùng làm background): {len(empty)}")
    return samples


# ---------------------------------------------------------------
# 2. Split train/val (theo ảnh gốc, trước khi convert -> không rò rỉ dữ liệu)
# ---------------------------------------------------------------
def split_samples(samples):
    present = {l for s in samples for l, _ in s["objects"]}
    n_val = max(1, round(len(samples) * VAL_RATIO))
    first = None
    for attempt in range(200):
        rng = random.Random(SEED + attempt)
        shuf = samples[:]
        rng.shuffle(shuf)
        val, train = shuf[:n_val], shuf[n_val:]
        if first is None:
            first = (train, val)
        v = {l for s in val for l, _ in s["objects"]}
        t = {l for s in train for l, _ in s["objects"]}
        if v == present and t == present:
            print(f"Split OK (seed={SEED + attempt}): mọi class đều có ở cả train và val")
            return train, val
    print("[WARN] Không tìm được split chứa đủ class ở cả 2 tập, dùng split ngẫu nhiên đầu tiên")
    return first


# ---------------------------------------------------------------
# 3. Letterbox + convert
# ---------------------------------------------------------------
def letterbox(img, size, color):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = cv2.copyMakeBorder(
        resized, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=color
    )
    return out, r, left, top


def convert_sample(sample, split, class_to_id):
    img = cv2.imread(str(sample["img"]))
    if img is None:
        raise RuntimeError(f"Không đọc được ảnh: {sample['img']}")
    h, w = img.shape[:2]

    # Toạ độ annotation theo kích thước khai báo trong file; nếu lệch ảnh thật thì scale lại
    sx = w / sample["dw"] if sample["dw"] else 1.0
    sy = h / sample["dh"] if sample["dh"] else 1.0
    if sample["dw"] and (sample["dw"], sample["dh"]) != (w, h):
        print(f"  [WARN] {sample['stem']}: ảnh {w}x{h} khác kích thước trong annotation "
              f"{sample['dw']}x{sample['dh']} -> scale toạ độ")

    if PRE_LETTERBOX:
        out_img, r, px, py = letterbox(img, IMG_SIZE, PAD_COLOR)
        W = H = IMG_SIZE
    else:
        out_img, r, px, py = img, 1.0, 0, 0
        W, H = w, h

    lines = []
    for label, pts in sample["objects"]:
        cid = class_to_id[label]
        p = np.array(pts, dtype=np.float64)
        p[:, 0] = np.clip(p[:, 0] * sx * r + px, 0, W) / W
        p[:, 1] = np.clip(p[:, 1] * sy * r + py, 0, H) / H
        if TASK == "detect":
            x1, y1 = p.min(0)
            x2, y2 = p.max(0)
            bw, bh = x2 - x1, y2 - y1
            if bw * W < 1 or bh * H < 1:
                print(f"  [WARN] {sample['stem']}: box '{label}' quá nhỏ sau convert -> bỏ")
                continue
            lines.append(f"{cid} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {bw:.6f} {bh:.6f}")
        else:
            lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in p.flatten()))

    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split
    if PRE_LETTERBOX:
        cv2.imwrite(str(img_dir / f"{sample['stem']}.png"), out_img)
    else:
        shutil.copy2(sample["img"], img_dir / sample["img"].name)
    (lbl_dir / f"{sample['stem']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


# ---------------------------------------------------------------
# 4. Debug: kiểm tra dataset sau khi convert (đọc lại từ file đã ghi)
# ---------------------------------------------------------------
def draw_labels(img, label_path, class_names):
    H, W = img.shape[:2]
    out = img.copy()
    thick = max(2, W // 600)
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        vals = np.array(parts[1:], dtype=float)
        color = color_of(cid)
        if TASK == "detect" and len(vals) == 4:
            cx, cy, bw, bh = vals
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            tl = (x1, y1)
        else:
            pts = (vals.reshape(-1, 2) * [W, H]).astype(np.int32)
            cv2.polylines(out, [pts], True, color, thick)
            tl = (int(pts[:, 0].min()), int(pts[:, 1].min()))
        cv2.putText(out, class_names[cid], (tl[0], max(20, tl[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 1600), color, thick)
    return out


def make_montage(imgs, cols=4, tile=480):
    tiles = []
    for im in imgs:
        s = tile / max(im.shape[:2])
        im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)))
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[:im.shape[0], :im.shape[1]] = im
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def verify_and_visualize(class_names):
    section("[DEBUG 3] Kiểm tra dataset YOLO đã ghi ra đĩa")
    nc = len(class_names)
    issues = 0
    for split in ("train", "val"):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        cnt = Counter()
        for ip in images:
            lp = lbl_dir / f"{ip.stem}.txt"
            if not lp.exists():
                print(f"  [ERR] thiếu label: {lp.name}")
                issues += 1
                continue
            for i, line in enumerate(lp.read_text().splitlines(), 1):
                parts = line.split()
                vals = [float(x) for x in parts[1:]]
                ok = int(parts[0]) < nc and all(0.0 <= v <= 1.0 for v in vals)
                ok &= (len(vals) == 4) if TASK == "detect" else (len(vals) >= 6 and len(vals) % 2 == 0)
                if not ok:
                    print(f"  [ERR] {lp.name}:{i} dòng không hợp lệ: {line[:80]}")
                    issues += 1
                else:
                    cnt[class_names[int(parts[0])]] += 1
        print(f"{split:5s}: {len(images)} ảnh | instances: {dict(cnt)}")
        for n in class_names:
            if cnt[n] == 0:
                print(f"  [WARN] class '{n}' không có instance nào trong {split}")

        # Vẽ lại label từ file .txt lên ảnh đã letterbox -> xác nhận toạ độ đúng
        out_dir = DEBUG_DIR / "labels_check" / split
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        drawn = []
        for ip in images:
            im = cv2.imread(str(ip))
            d = draw_labels(im, lbl_dir / f"{ip.stem}.txt", class_names)
            cv2.imwrite(str(out_dir / f"{ip.stem}.jpg"), d, [cv2.IMWRITE_JPEG_QUALITY, 85])
            drawn.append(d)
        if drawn:
            cv2.imwrite(str(DEBUG_DIR / f"montage_{split}.jpg"), make_montage(drawn[:12]))
        print(f"  -> ảnh kiểm tra: {out_dir}")
        print(f"  -> montage:      {DEBUG_DIR / f'montage_{split}.jpg'}")
    print(f"\nTổng số lỗi: {issues}")
    if issues:
        raise RuntimeError("Dataset có lỗi, xem log phía trên.")
    print("MỞ montage_train.jpg / montage_val.jpg để xác nhận box/polygon nằm đúng vật thể trước khi train.")


# ---------------------------------------------------------------
# 5. Prepare
# ---------------------------------------------------------------
def prepare_dataset():
    samples = load_samples()
    labels = sorted({l for s in samples for l, _ in s["objects"]})
    if CLASS_NAMES is None:
        class_names = labels
    else:
        unknown = set(labels) - set(CLASS_NAMES)
        if unknown:
            raise RuntimeError(f"Label không có trong CLASS_NAMES: {unknown}")
        class_names = list(CLASS_NAMES)
    class_to_id = {n: i for i, n in enumerate(class_names)}

    section("[DEBUG 2] Class mapping & split")
    for n, i in class_to_id.items():
        print(f"  {i}: {n}")

    train, val = split_samples(samples)
    print(f"Train: {len(train)} ảnh | Val: {len(val)} ảnh")

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for sp in ("train", "val"):
        (DATASET_DIR / "images" / sp).mkdir(parents=True)
        (DATASET_DIR / "labels" / sp).mkdir(parents=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for split, group in (("train", train), ("val", val)):
        for s in group:
            n = convert_sample(s, split, class_to_id)
            manifest[s["stem"]] = dict(split=split, raw=str(s["img"]), n_objects=n)
    (DATASET_DIR / "split.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Danh sách split: {DATASET_DIR / 'split.json'}")

    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    (DATASET_DIR / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"data.yaml: {DATASET_DIR / 'data.yaml'}")

    verify_and_visualize(class_names)


# ---------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------
def run_predictions(model, paths, out_dir, **kw):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.predict(str(paths[0]), imgsz=IMG_SIZE, verbose=False, **kw)  # warmup
    rows = []
    for p in paths:
        t = time.perf_counter()
        r = model.predict(str(p), imgsz=IMG_SIZE, conf=PREDICT_CONF, verbose=False, **kw)[0]
        dt = (time.perf_counter() - t) * 1000
        cv2.imwrite(str(out_dir / f"{p.stem}.jpg"), r.plot())
        rows.append((p.name, len(r.boxes), dt))
    return rows


def train(yaml_path):
    from ultralytics import YOLO

    section("[TRAIN] YOLOv8 trên CPU")
    print(f"Weights: {MODEL_WEIGHTS} | imgsz={IMG_SIZE} | batch={BATCH} | epochs={EPOCHS}")
    print("(CPU + imgsz 1280: mỗi epoch có thể mất vài chục giây tới vài phút, hãy kiên nhẫn)")
    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",
        workers=WORKERS,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        patience=PATIENCE,
        cache=CACHE,
        seed=SEED,
        plots=True,
    )
    best = Path(model.trainer.best)
    print(f"\nBest weights: {best}")
    print(f"Biểu đồ loss/metric (results.png, confusion_matrix.png, val_batch*_pred.jpg): {best.parent.parent}")

    section("[DEBUG 4] Đánh giá best.pt trên tập val")
    best_model = YOLO(str(best))
    m = best_model.val(data=str(yaml_path), imgsz=IMG_SIZE, device="cpu", plots=False, verbose=False)
    res = m.seg if TASK == "segment" else m.box
    print(f"Precision={res.mp:.3f}  Recall={res.mr:.3f}  mAP50={res.map50:.3f}  mAP50-95={res.map:.3f}")
    print("(tập val chỉ ~10 ảnh nên số liệu dao động mạnh, hãy nhìn cả ảnh dự đoán)")
    return best


# ---------------------------------------------------------------
# 7. Export OpenVINO + so sánh
# ---------------------------------------------------------------
def export_openvino(best_pt, yaml_path):
    from ultralytics import YOLO

    section("[EXPORT] OpenVINO")
    if not best_pt.exists():
        raise FileNotFoundError(f"Không thấy {best_pt}")
    pt_model = YOLO(str(best_pt))
    ov_dir = pt_model.export(
        format="openvino",
        imgsz=IMG_SIZE,
        half=False,
        int8=EXPORT_INT8,
        data=str(yaml_path),   # dùng để calibrate khi int8
        dynamic=False,
        batch=1,
    )
    print(f"Đã export: {ov_dir}")
    ov_model = YOLO(str(ov_dir), task=TASK)

    section("[DEBUG 5] So sánh PyTorch vs OpenVINO trên ảnh val (CPU)")
    manifest = json.loads((DATASET_DIR / "split.json").read_text())
    val_imgs = sorted(p for p in (DATASET_DIR / "images" / "val").iterdir() if p.suffix.lower() in IMG_EXTS)
    rows_pt = run_predictions(pt_model, val_imgs, DEBUG_DIR / "pred_pytorch", device="cpu")
    rows_ov = run_predictions(ov_model, val_imgs, DEBUG_DIR / "pred_openvino")
    print(f"{'ảnh':40s} {'PT #det':>8s} {'OV #det':>8s} {'PT ms':>8s} {'OV ms':>8s}")
    for (n, c1, t1), (_, c2, t2) in zip(rows_pt, rows_ov):
        flag = "" if c1 == c2 else "  <-- khác số lượng"
        print(f"{n:40s} {c1:8d} {c2:8d} {t1:8.1f} {t2:8.1f}{flag}")
    avg_pt = np.mean([r[2] for r in rows_pt])
    avg_ov = np.mean([r[2] for r in rows_ov])
    print(f"\nTrung bình: PyTorch {avg_pt:.1f} ms | OpenVINO {avg_ov:.1f} ms | tăng tốc x{avg_pt / avg_ov:.2f}")
    print(f"Ảnh dự đoán: {DEBUG_DIR / 'pred_pytorch'}  và  {DEBUG_DIR / 'pred_openvino'}")

    # Test trên ảnh GỐC 2688x1520 (giống lúc chạy thật, Ultralytics tự letterbox)
    section("[DEBUG 6] OpenVINO trên ảnh gốc 2688x1520 (mô phỏng deploy)")
    raw_dir = DEBUG_DIR / "pred_openvino_raw"
    raw_paths = [Path(v["raw"]) for v in manifest.values() if v["split"] == "val"][:5]
    for n, c, t in run_predictions(ov_model, raw_paths, raw_dir):
        print(f"{n:40s} #det={c:3d}  {t:7.1f} ms")
    print(f"Ảnh: {raw_dir}")
    print(f"\nModel OpenVINO để dùng khi deploy: {ov_dir}")
    print('Load:  YOLO("<thư mục>_openvino_model", task="%s").predict(img, imgsz=%d)' % (TASK, IMG_SIZE))


def main():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = DATASET_DIR / "data.yaml"

    if STEP_PREPARE:
        prepare_dataset()
    if (STEP_TRAIN or STEP_EXPORT) and not yaml_path.exists():
        raise FileNotFoundError("Chưa có dataset, hãy bật STEP_PREPARE = True")

    best = None
    if STEP_TRAIN:
        best = train(yaml_path)
    if STEP_EXPORT:
        best = best or (RUNS_DIR / RUN_NAME / "weights" / "best.pt")
        export_openvino(best, yaml_path)


if __name__ == "__main__":  # bắt buộc trên Windows vì dataloader dùng multiprocessing
    main()