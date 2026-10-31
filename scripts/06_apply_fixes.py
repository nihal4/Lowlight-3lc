"""Step 6 - apply the label fixes you ACCEPTED in the review queue.

Only rows with status == accepted are applied (case-insensitive). Default is a
dry run that prints what would change; add --apply to write. Before writing,
the current labels are backed up to <work>/dataset/labels_backup_<timestamp>/.

  missing_label     -> append pred box
  class_flip        -> change class of the label at gt_index to pred_cls
  loose_box         -> replace label at gt_index with pred box (keeps label class)
  possible_spurious -> delete label at gt_index

Typical Kaggle flow: download outputs/review/review_queue.csv, mark rows, upload
it as a Kaggle Dataset, then run this with --queue /kaggle/input/<ds>/review_queue.csv.
"""
import argparse
import csv
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from lowlight.common import CLASSES, banner, dataset_dir, outputs_dir, read_labels, write_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=None, help="default: outputs/review/review_queue.csv")
    ap.add_argument("--apply", action="store_true", help="write changes (otherwise dry run)")
    args = ap.parse_args()

    banner("Step 6: apply accepted label fixes" + ("" if args.apply else " (DRY RUN)"))
    qpath = Path(args.queue) if args.queue else outputs_dir() / "review" / "review_queue.csv"
    if not qpath.exists():
        raise SystemExit(f"{qpath} not found - run scripts/05_mine_labels.py first")
    rows = list(csv.DictReader(open(qpath)))
    accepted = [r for r in rows if r.get("status", "").strip().lower() == "accepted"]
    print(f"{len(rows)} rows in queue, {len(accepted)} accepted: {dict(Counter(r['type'] for r in accepted))}")
    if not accepted:
        print("Nothing to apply.")
        return

    cls_id = {n: i for i, n in enumerate(CLASSES)}
    ds = dataset_dir()
    by_file = defaultdict(list)
    for r in accepted:
        by_file[(r["split"], Path(r["image"]).stem)].append(r)

    if args.apply:
        backup = ds / f"labels_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(ds / "labels", backup, ignore=shutil.ignore_patterns("*.cache"))
        print(f"Backup: {backup}")

    changed = Counter()
    for (split, stem), items in by_file.items():
        path = ds / "labels" / split / f"{stem}.txt"
        boxes = read_labels(path)
        n0 = len(boxes)
        delete, add = set(), []
        for r in items:
            kind = r["type"]
            gi = int(r["gt_index"]) if r["gt_index"] != "" else None
            pred = [float(v) for v in r["pred_box"].split()] if r["pred_box"] else None
            if gi is not None and gi >= n0:
                print(f"  skip #{r['qid']}: gt_index {gi} no longer exists in {path.name}")
                continue
            if kind == "missing_label":
                add.append([cls_id[r["pred_cls"]], *pred])
            elif kind == "class_flip":
                boxes[gi][0] = cls_id[r["pred_cls"]]
            elif kind == "loose_box":
                boxes[gi] = [boxes[gi][0], *pred]
            elif kind == "possible_spurious":
                delete.add(gi)
            else:
                print(f"  skip #{r['qid']}: unknown type {kind}")
                continue
            changed[kind] += 1
        boxes = [b for i, b in enumerate(boxes) if i not in delete] + add
        if args.apply:
            write_labels(path, boxes)
        print(f"  {split}/{stem}.txt: {n0} -> {len(boxes)} boxes")

    if args.apply:
        for c in (ds / "labels").glob("*.cache"):
            c.unlink()
    print(f"\n{'Applied' if args.apply else 'Would apply'}: {dict(changed)}")
    if not args.apply:
        print("Re-run with --apply to write these changes.")
    else:
        print("Retrain next, e.g.: python scripts/04_train.py --name v2_fixed")


if __name__ == "__main__":
    main()
