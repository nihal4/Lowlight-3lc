"""Step 4 - train YOLOv11n under the locked competition contract, on CUDA.

  * model  : official yolo11n.pt, downloaded once and sha256-verified
  * imgsz  : 640, pretrained=True (locked; the script refuses to change them)
  * device : all visible GPUs (DDP on Kaggle T4 x2), AMP on
  * output : <work>/runs/<name>/weights/best.pt, copied to outputs/best_<name>.pt,
             plus provenance.json recording checkpoint hash + every train arg

Examples
  python scripts/04_train.py                          # config defaults
  python scripts/04_train.py --epochs 2 --name smoke  # quick check
  python scripts/04_train.py --name v2_fixed --set mixup=0.0 --set lr0=0.005
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from lowlight.common import (IMGSZ, LOCKED_KEYS, OFFICIAL_CHECKPOINT_SHA256, REPO_ROOT, TRAIN_BOUNDS,
                             banner, dataset_dir, ensure_official_checkpoint, outputs_dir,
                             resolve_device, runs_dir, save_json, sha256_file)


def parse_value(raw: str):
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for conv in (int, float):
        try:
            return conv(raw)
        except ValueError:
            pass
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "train.yaml"))
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | 0,1")
    ap.add_argument("--data", default=None, help="dataset.yaml (default: working dataset)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any train arg")
    args = ap.parse_args()

    banner(f"Step 4: train '{args.name}'")
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        cfg[k.strip()] = parse_value(v.strip())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch

    locked = LOCKED_KEYS & set(cfg)
    if locked:
        raise SystemExit(f"These are locked by the competition contract and can't be set: {sorted(locked)}")
    for key, (lo, hi) in TRAIN_BOUNDS.items():
        if key in cfg and not lo <= cfg[key] <= hi:
            raise SystemExit(f"{key} must be between {lo} and {hi} (got {cfg[key]})")

    data = Path(args.data) if args.data else dataset_dir() / "dataset.yaml"
    if not data.exists():
        raise SystemExit(f"{data} not found - run scripts/02_prepare_dataset.py first")

    device = resolve_device(args.device)
    if device == "cpu":
        cfg["amp"] = False
        cfg["workers"] = min(int(cfg.get("workers", 0)), 2)
    if isinstance(device, list) and int(cfg.get("batch", 16)) % len(device):
        raise SystemExit(f"batch ({cfg['batch']}) must be divisible by the number of GPUs ({len(device)})")

    ckpt = ensure_official_checkpoint()
    ckpt_sha = sha256_file(ckpt)
    assert ckpt_sha == OFFICIAL_CHECKPOINT_SHA256

    from ultralytics import YOLO
    import ultralytics

    train_kwargs = dict(cfg)
    train_kwargs.update(
        data=str(data),
        imgsz=IMGSZ,          # locked
        pretrained=True,      # locked
        device=device,
        project=str(runs_dir()),
        name=args.name,
        exist_ok=True,
    )
    print(f"checkpoint : {ckpt} (sha256 {ckpt_sha[:12]}...)")
    print(f"device     : {device}")
    print("train args : " + ", ".join(f"{k}={v}" for k, v in sorted(train_kwargs.items())))

    t0 = time.time()
    model = YOLO(str(ckpt))
    model.train(**train_kwargs)
    minutes = (time.time() - t0) / 60

    run_dir = runs_dir() / args.name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but {best} is missing")
    out_best = outputs_dir() / f"best_{args.name}.pt"
    shutil.copy2(best, out_best)

    # Validation metrics of the best checkpoint (single device, same imgsz).
    vdev = device[0] if isinstance(device, list) else device
    metrics = YOLO(str(best)).val(data=str(data), imgsz=IMGSZ, device=vdev, batch=16,
                                  plots=False, verbose=False, project=str(run_dir), name="val_best",
                                  exist_ok=True)
    summary = {
        "run": args.name,
        "minutes": round(minutes, 1),
        "val_mAP50": round(float(metrics.box.map50), 4),
        "val_mAP50_95": round(float(metrics.box.map), 4),
        "per_class_mAP50": {metrics.names[int(c)]: round(float(ap), 4)
                            for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)},
    }
    provenance = {
        "model": "yolo11n.pt", "checkpoint_sha256": ckpt_sha, "imgsz": IMGSZ, "pretrained": True,
        "ultralytics": ultralytics.__version__, "train_args": {k: str(v) for k, v in train_kwargs.items()},
        "results": summary,
    }
    save_json(provenance, run_dir / "provenance.json")
    save_json(provenance, outputs_dir() / f"provenance_{args.name}.json")

    print(json.dumps(summary, indent=2))
    print(f"\nBest weights: {best}\nCopied to   : {out_best}")


if __name__ == "__main__":
    main()
