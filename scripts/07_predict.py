"""Step 7 - predict the test set and write submission.csv.

Format (validated before saving, same rules as the competition metric):
  id,image_id,prediction_string
  0,2015_00015,"10 0.91234 0.51200 0.40100 0.21000 0.55000 4 0.40000 ..."
  * one row per test image (715), id = 0..714 aligned to image_id sorted ascending
  * prediction_string = groups of 'class conf x_center y_center width height',
    YOLO-normalized to [0,1]; 'no box' when there are no detections

mAP rewards recall, so the default confidence floor is low (0.001) with up to
100 boxes per image (COCO mAP only counts the top 100 per image anyway).
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from lowlight.common import (COMPETITION_SLUG, IMGSZ, ON_KAGGLE, banner, cuda_devices, dataset_dir,
                             kit_dir, outputs_dir, runs_dir, save_json)
from lowlight.submission import build_df, format_boxes, validate_df

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_sample_submission():
    cands = [kit_dir() / "sample_submission.csv"]
    if ON_KAGGLE:
        cands += sorted(Path("/kaggle/input").glob("**/sample_submission.csv"))
        cands.insert(0, Path(f"/kaggle/input/{COMPETITION_SLUG}/sample_submission.csv"))
    return next((p for p in cands if p.exists()), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="default: <work>/runs/baseline/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tta", action="store_true",
                    help="ultralytics test-time augmentation (flip + scales). Check the Rules before submitting with it.")
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--out", default=None, help="default: outputs/submission.csv")
    args = ap.parse_args()

    banner("Step 7: predict test set -> submission.csv")
    from ultralytics import YOLO

    weights = Path(args.weights) if args.weights else runs_dir() / "baseline" / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} not found")
    test_dir = Path(args.test_dir) if args.test_dir else dataset_dir() / "images" / "test"
    imgs = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    gpus = cuda_devices()
    device = gpus[0] if gpus else "cpu"
    print(f"weights {weights}\n{len(imgs)} test images, device={device}, conf={args.conf}, "
          f"max_det={args.max_det}, tta={args.tta}")

    model = YOLO(str(weights))
    pred_map = {}
    for start in range(0, len(imgs), args.batch):
        chunk = imgs[start:start + args.batch]
        results = model.predict([str(p) for p in chunk], imgsz=IMGSZ, conf=args.conf, iou=args.iou,
                                max_det=args.max_det, device=device, half=bool(gpus),
                                augment=args.tta, verbose=False)
        for img, res in zip(chunk, results):
            b = res.boxes
            pred_map[img.stem] = ("no box" if b is None or not len(b)
                                  else format_boxes(b.cls.tolist(), b.conf.tolist(), b.xywhn.tolist()))
        print(f"  {min(start + args.batch, len(imgs))}/{len(imgs)}", end="\r", flush=True)
    print()

    df = build_df(pred_map)
    sample = find_sample_submission()
    sample_ids = None
    if sample is not None:
        import pandas as pd

        sample_ids = pd.read_csv(sample)["image_id"].astype(str).tolist()
        print(f"Checking image ids against {sample}")
    expected = len(sample_ids) if sample_ids is not None else (715 if len(imgs) == 715 else None)
    summary = validate_df(df, expected_rows=expected, sample_ids=sample_ids)

    out = Path(args.out) if args.out else outputs_dir() / "submission.csv"
    df.to_csv(out, index=False)
    save_json({"weights": str(weights), **vars(args), **summary}, out.with_suffix(".summary.json"))
    print(f"Validation passed: {summary['rows']} rows, {summary['boxes']} boxes "
          f"({summary['boxes_per_image']}/image), {summary['empty_images']} empty")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
