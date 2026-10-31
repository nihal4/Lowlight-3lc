"""Step 2 - build the working dataset the model trains on.

Images are symlinked from the kit (no copy, no extra disk); labels are COPIED so
label fixes (step 6) never modify the original kit. If the working dataset
already exists it is kept, so your fixes survive re-runs. Use --reset to start
over from the kit's original labels.
"""
import argparse
import os
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from lowlight.common import banner, dataset_dir, kit_dir, write_dataset_yaml


def link_or_copy_dir(src: Path, dst: Path) -> None:
    """Populate dst with one entry per image: hardlink (no extra disk), else
    copy. NOT a directory symlink: Ultralytics resolves symlinked folders and
    would then read labels from the original kit, silently ignoring fixes."""
    if dst.is_symlink():
        dst.unlink()  # left over from an older version of this script
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if target.exists():
            continue
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="discard label fixes, re-copy original labels")
    args = ap.parse_args()

    banner("Step 2: prepare working dataset")
    kit = kit_dir()
    if not (kit / "images" / "train").is_dir():
        raise SystemExit(f"Kit not found at {kit}. Run scripts/01_download_kit.py first "
                         f"(or set LOWLIGHT_KIT).")
    ds = dataset_dir()
    if args.reset and ds.exists():
        print(f"Resetting {ds}")
        shutil.rmtree(ds)

    for split in ("train", "val", "test"):
        src = kit / "images" / split
        if src.is_dir():
            link_or_copy_dir(src, ds / "images" / split)
    for split in ("train", "val"):
        dst = ds / "labels" / split
        if not dst.exists():
            shutil.copytree(kit / "labels" / split, dst)
            print(f"  copied labels/{split}")
        else:
            print(f"  kept existing labels/{split} (use --reset to restore originals)")
    originals = ds / "labels_original"
    if not originals.exists():
        shutil.copytree(kit / "labels", originals)

    y = write_dataset_yaml(ds)
    # Ultralytics writes *.cache files next to labels; drop stale ones.
    for c in (ds / "labels").glob("*.cache"):
        c.unlink()
    for split in ("train", "val", "test"):
        n = len(list((ds / "images" / split).glob("*"))) if (ds / "images" / split).exists() else 0
        print(f"  {split:5s}: {n} images")
    print(f"dataset.yaml -> {y}")


if __name__ == "__main__":
    main()
