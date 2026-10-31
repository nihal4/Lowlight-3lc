"""Step 5 - find likely label errors using the trained model (the data-centric loop).

Runs the model over train (and val) and compares predictions with the labels:

  missing_label     model is confident, no label anywhere near it      -> add box
  class_flip        model's box sits on a label of a DIFFERENT class   -> change class
  loose_box         same class, but the label box fits poorly          -> replace box
  possible_spurious a label the model never finds, even at low conf    -> remove box

Nothing is changed here. It writes a REVIEW QUEUE:
  outputs/review/review_queue.csv   one row per candidate, empty `status` column
  outputs/review/previews/*.jpg     CLAHE-brightened image: green = label, red = model

Look at each preview and set status to `accepted` or `rejected`, then run
06_apply_fixes.py. Only accepted rows are applied. Don't bulk-accept: the
model is wrong often enough on dark images that unreviewed edits hurt.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from lowlight.common import (CLASSES, IMGSZ, banner, cuda_devices, dataset_dir, iou_xywh,
                             outputs_dir, read_labels, runs_dir)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIELDS = ["qid", "split", "image", "type", "gt_index", "gt_cls", "gt_box", "pred_cls", "pred_conf",
          "pred_box", "iou", "score", "preview", "status", "note"]


def fmt_box(b):
    return " ".join(f"{v:.5f}" for v in b) if b is not None else ""


def draw_preview(img_path: Path, gt, pred, gt_idx, out_path: Path, title: str) -> None:
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    h, w = img.shape[:2]

    def rect(b, color, label, thick, below=False):
        x1, y1 = int((b[0] - b[2] / 2) * w), int((b[1] - b[3] / 2) * h)
        x2, y2 = int((b[0] + b[2] / 2) * w), int((b[1] + b[3] / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        ty = min(y2 + 14, h - 22) if below else max(y1 - 4, 12)
        cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for i, g in enumerate(gt):
        rect(g[1:], (0, 255, 0), "label: " + CLASSES[int(g[0])], 3 if i == gt_idx else 1)
    if pred is not None:
        rect(pred[2:], (0, 0, 255), f"model: {CLASSES[int(pred[0])]} {pred[1]:.2f}", 2, below=True)
    cv2.putText(img, title, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    scale = 800 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="default: <work>/runs/baseline/weights/best.pt")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--high-conf", type=float, default=0.6, help="confidence for 'model is sure'")
    ap.add_argument("--low-conf", type=float, default=0.05, help="floor for 'model sees anything'")
    ap.add_argument("--max-previews", type=int, default=150, help="preview images per type")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    banner("Step 5: mine likely label errors")
    from ultralytics import YOLO

    weights = Path(args.weights) if args.weights else runs_dir() / "baseline" / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} not found - train first (scripts/04_train.py)")
    gpus = cuda_devices()
    device = gpus[0] if gpus else "cpu"
    model = YOLO(str(weights))
    ds = dataset_dir()
    review = outputs_dir() / "review"
    rows = []

    for split in args.splits.split(","):
        imgs = sorted(p for p in (ds / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS)
        print(f"[{split}] scanning {len(imgs)} images on {device} ...")
        for start in range(0, len(imgs), args.batch):
            chunk = imgs[start:start + args.batch]
            results = model.predict([str(p) for p in chunk], imgsz=IMGSZ, conf=args.low_conf, iou=0.6,
                                    device=device, half=bool(gpus), verbose=False, max_det=100)
            for img, res in zip(chunk, results):
                gt = read_labels(ds / "labels" / split / (img.stem + ".txt"))
                preds = []
                if res.boxes is not None and len(res.boxes):
                    for c, cf, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xywhn.tolist()):
                        preds.append([int(c), float(cf), *b])
                flipped = set()

                def add(kind, gi, p, iou, score):
                    rows.append({
                        "split": split, "image": img.name, "type": kind,
                        "gt_index": "" if gi is None else gi,
                        "gt_cls": "" if gi is None else CLASSES[int(gt[gi][0])],
                        "gt_box": "" if gi is None else fmt_box(gt[gi][1:]),
                        "pred_cls": "" if p is None else CLASSES[p[0]],
                        "pred_conf": "" if p is None else round(p[1], 4),
                        "pred_box": "" if p is None else fmt_box(p[2:]),
                        "iou": round(iou, 3), "score": round(score, 4), "status": "", "note": "",
                        "_img": img, "_gt": gt, "_pred": p, "_gi": gi,
                    })

                for p in preds:
                    if p[1] < args.high_conf:
                        continue
                    ious = [iou_xywh(p[2:], g[1:]) for g in gt]
                    any_best = max(ious, default=0.0)
                    same = [(iou, i) for i, iou in enumerate(ious) if int(gt[i][0]) == p[0]]
                    same_best, same_i = max(same, default=(0.0, None))
                    if any_best < 0.1:
                        add("missing_label", None, p, any_best, p[1])
                    elif same_best < 0.3:
                        oi = max(range(len(ious)), key=lambda i: ious[i])
                        if ious[oi] >= 0.5 and int(gt[oi][0]) != p[0] and oi not in flipped:
                            flipped.add(oi)
                            add("class_flip", oi, p, ious[oi], p[1])
                    elif same_best < 0.5:
                        add("loose_box", same_i, p, same_best, p[1] * (1 - same_best))

                for gi, g in enumerate(gt):
                    same_best = max((iou_xywh(p[2:], g[1:]) for p in preds if p[0] == int(g[0])), default=0.0)
                    any_best = max((iou_xywh(p[2:], g[1:]) for p in preds), default=0.0)
                    if same_best < 0.1 and any_best < 0.5:
                        add("possible_spurious", gi, None, same_best, 1.0 - any_best)

    # Rank by score within each type and render previews for the top ones.
    rows.sort(key=lambda r: (r["type"], -r["score"]))
    shown = Counter()
    for qid, r in enumerate(rows):
        r["qid"] = qid
        if shown[r["type"]] < args.max_previews:
            shown[r["type"]] += 1
            name = f"{r['type']}/{qid:05d}_{Path(r['image']).stem}.jpg"
            title = f"#{qid} {r['type']}"
            draw_preview(r["_img"], r["_gt"], r["_pred"], r["_gi"], review / "previews" / name, title)
            r["preview"] = f"previews/{name}"
        else:
            r["preview"] = ""

    review.mkdir(parents=True, exist_ok=True)
    out_csv = review / "review_queue.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["type"] for r in rows)
    print(f"\n{len(rows)} candidates: {dict(counts)}")
    print(f"Queue   : {out_csv}")
    print(f"Previews: {review / 'previews'}  (green = label, red = model, thick green = the label in question)")
    print("Next: set status=accepted/rejected per row, then run scripts/06_apply_fixes.py")


if __name__ == "__main__":
    main()
