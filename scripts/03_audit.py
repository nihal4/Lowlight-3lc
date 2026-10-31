"""Step 3 - data audit: class balance, box sizes, darkness, and label sanity.

Writes outputs/audit/audit.json and a few plots. Label problems that can be
detected without a model (out-of-range coordinates, invalid class ids,
zero-size boxes, exact duplicates, empty label files) are listed so you know
what you're training on before the first epoch.
"""
import argparse
import random
from collections import Counter

import _bootstrap  # noqa: F401
from lowlight.common import CLASSES, NUM_CLASSES, banner, dataset_dir, outputs_dir, save_json

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brightness-sample", type=int, default=600, help="images per split for luminance stats")
    args = ap.parse_args()

    banner("Step 3: data audit")
    import numpy as np
    from PIL import Image

    ds = dataset_dir()
    out = outputs_dir() / "audit"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    issues = []

    for split in ("train", "val"):
        imgs = sorted(p for p in (ds / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS)
        box_counts = Counter()
        img_counts = Counter()
        areas = {c: [] for c in range(NUM_CLASSES)}
        empty = 0
        for img in imgs:
            lbl = ds / "labels" / split / (img.stem + ".txt")
            lines = lbl.read_text().splitlines() if lbl.exists() else []
            lines = [l for l in lines if l.strip()]
            if not lines:
                empty += 1
            seen, classes_here = set(), set()
            for ln, line in enumerate(lines):
                p = line.split()
                try:
                    c, xc, yc, w, h = int(float(p[0])), *map(float, p[1:5])
                except (ValueError, IndexError):
                    issues.append({"split": split, "image": img.name, "line": ln, "issue": "unparsable"})
                    continue
                if not 0 <= c < NUM_CLASSES:
                    issues.append({"split": split, "image": img.name, "line": ln, "issue": f"class {c} out of range"})
                    continue
                if not all(0 <= v <= 1 for v in (xc, yc, w, h)):
                    issues.append({"split": split, "image": img.name, "line": ln, "issue": "coords outside [0,1]"})
                if w <= 0.002 or h <= 0.002:
                    issues.append({"split": split, "image": img.name, "line": ln, "issue": "degenerate (tiny) box"})
                key = tuple(round(v, 4) for v in (c, xc, yc, w, h))
                if key in seen:
                    issues.append({"split": split, "image": img.name, "line": ln, "issue": "exact duplicate box"})
                seen.add(key)
                box_counts[c] += 1
                classes_here.add(c)
                areas[c].append(w * h)
            for c in classes_here:
                img_counts[c] += 1

        lum = []
        for img in random.Random(0).sample(imgs, min(args.brightness_sample, len(imgs))):
            with Image.open(img) as im:
                g = np.asarray(im.convert("L").resize((128, 128)), dtype=np.float32)
            lum.append(float(g.mean()))
        lum = np.array(lum) if lum else np.zeros(1)

        report[split] = {
            "images": len(imgs),
            "empty_label_images": empty,
            "boxes_per_class": {CLASSES[c]: box_counts[c] for c in range(NUM_CLASSES)},
            "images_per_class": {CLASSES[c]: img_counts[c] for c in range(NUM_CLASSES)},
            "median_box_area": {CLASSES[c]: float(np.median(areas[c])) if areas[c] else None
                                for c in range(NUM_CLASSES)},
            "luminance": {"mean": float(lum.mean()), "p10": float(np.percentile(lum, 10)),
                          "p50": float(np.percentile(lum, 50)), "p90": float(np.percentile(lum, 90)),
                          "share_below_40": float((lum < 40).mean())},
        }
        r = report[split]
        print(f"\n[{split}] {r['images']} images, {sum(box_counts.values())} boxes, {empty} with no labels")
        print(f"  luminance mean {r['luminance']['mean']:.1f}/255, "
              f"{100 * r['luminance']['share_below_40']:.0f}% of images very dark (<40)")
        for c in range(NUM_CLASSES):
            print(f"  {c:2d} {CLASSES[c]:10s} boxes={box_counts[c]:5d}  images={img_counts[c]:5d}")

    report["label_issues"] = {"count": len(issues), "by_type": dict(Counter(i["issue"] for i in issues)),
                              "examples": issues[:200]}
    print(f"\nLabel sanity issues: {len(issues)} {dict(Counter(i['issue'] for i in issues))}")
    save_json(report, out / "audit.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(NUM_CLASSES)
        ax.bar(x - 0.2, [report["train"]["boxes_per_class"][c] for c in CLASSES], 0.4, label="train")
        ax.bar(x + 0.2, [report["val"]["boxes_per_class"][c] for c in CLASSES], 0.4, label="val")
        ax.set_xticks(x, CLASSES, rotation=45, ha="right")
        ax.set_ylabel("boxes")
        ax.set_title("Boxes per class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "class_balance.png", dpi=110)
        plt.close(fig)
        print(f"Plots + audit.json -> {out}")
    except Exception as exc:
        print(f"(plotting skipped: {exc})")


if __name__ == "__main__":
    main()
