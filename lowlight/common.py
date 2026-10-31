"""Shared constants, paths and helpers for the 3LC Low-Light Detection pipeline.

Everything that encodes the competition contract lives here so every script
agrees on it:
  * model      = official Ultralytics COCO-pretrained yolo11n.pt (sha256-pinned)
  * imgsz      = 640
  * pretrained = True
  * 12 classes, YOLO-normalized boxes
  * submission = id,image_id,prediction_string  (715 rows, sorted image_id)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

# ── Competition contract ────────────────────────────────────────────────
COMPETITION_SLUG = "the-3lc-low-light-detection-challenge"
CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
NUM_CLASSES = len(CLASSES)
IMGSZ = 640  # locked
EXPECTED_TEST_ROWS = 715
SUBMISSION_COLUMNS = ["id", "image_id", "prediction_string"]

OFFICIAL_CHECKPOINT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
OFFICIAL_CHECKPOINT_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"

# Bounds the official plugin enforces on training args (trainer.py _EXPOSED).
TRAIN_BOUNDS = {
    "epochs": (1, 300),
    "batch": (1, 128),
    "lr0": (0.0001, 0.1),
    "lrf": (0.01, 1.0),
    "patience": (0, 100),
}
LOCKED_KEYS = {"model", "imgsz", "pretrained", "weights", "resume"}

# Starter kit CDN (same source the 3LC Hub plugin downloads from).
CDN_BASE_URL = "https://competitions.3lc.ai"
KIT_VERSION = "v3"
KIT_PREFIX = os.environ.get(
    "LOWLIGHT_KIT_URL", f"{CDN_BASE_URL}/kaggle/exdark-low-light/starter-kit/{KIT_VERSION}"
).rstrip("/")

# ── Paths ───────────────────────────────────────────────────────────────
ON_KAGGLE = Path("/kaggle/working").is_dir()
REPO_ROOT = Path(__file__).resolve().parents[1]


def work_dir() -> Path:
    """Writable root for data, runs and outputs."""
    env = os.environ.get("LOWLIGHT_WORK")
    if env:
        p = Path(env)
    elif ON_KAGGLE:
        p = Path("/kaggle/working/lowlight")
    else:
        p = REPO_ROOT / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


def kit_dir() -> Path:
    """Folder holding the starter kit (dataset.yaml, images/, labels/)."""
    env = os.environ.get("LOWLIGHT_KIT")
    return Path(env) if env else work_dir() / "data" / KIT_VERSION / "starter_kit"


def dataset_dir() -> Path:
    """The dataset actually trained on. Starts as a copy of the kit's labels
    (images are symlinked) so label fixes never touch the original kit."""
    return work_dir() / "dataset"


def runs_dir() -> Path:
    p = work_dir() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir() -> Path:
    p = Path("/kaggle/working") if ON_KAGGLE else work_dir() / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Helpers ─────────────────────────────────────────────────────────────
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ensure_official_checkpoint() -> Path:
    """Download (once) and sha256-verify the official yolo11n.pt."""
    dest = work_dir() / "checkpoints" / "yolo11n.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == OFFICIAL_CHECKPOINT_SHA256:
        return dest
    tmp = dest.with_suffix(".part")
    print(f"Downloading official checkpoint: {OFFICIAL_CHECKPOINT_URL}")
    with urllib.request.urlopen(OFFICIAL_CHECKPOINT_URL, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    digest = sha256_file(tmp)
    if digest != OFFICIAL_CHECKPOINT_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"yolo11n.pt sha256 mismatch: got {digest[:12]}..., "
            f"expected {OFFICIAL_CHECKPOINT_SHA256[:12]}..."
        )
    tmp.replace(dest)
    print(f"Checkpoint verified (sha256 {digest[:12]}...)")
    return dest


def cuda_devices() -> list[int]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def resolve_device(raw: str | None = None):
    """'' / 'auto' -> all CUDA GPUs (DDP when >1), else CPU."""
    raw = (raw or "auto").strip().lower()
    if raw not in ("", "auto"):
        return raw
    gpus = cuda_devices()
    if not gpus:
        return "cpu"
    return gpus[0] if len(gpus) == 1 else gpus


def write_dataset_yaml(root: Path) -> Path:
    """Write a YOLO dataset.yaml with absolute paths (ultralytics mis-resolves
    relative 'path: .')."""
    yaml_path = root / "dataset.yaml"
    lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {NUM_CLASSES}",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def read_labels(path: Path) -> list[list[float]]:
    """YOLO label file -> [[cls, xc, yc, w, h], ...] (missing file = no boxes)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append([float(v) for v in parts[:5]])
    return out


def write_labels(path: Path, boxes: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in boxes)
    )


def iou_xywh(a, b) -> float:
    """IoU of two normalized (xc, yc, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70, flush=True)
