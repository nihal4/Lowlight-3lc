# 🌙 3LC Low-Light Detection Challenge — CUDA pipeline

Pipeline for [The 3LC Low-Light Detection Challenge](https://www.kaggle.com/competitions/the-3lc-low-light-detection-challenge):
detect 12 everyday object classes in low-light photos (ExDark), scored by **COCO mAP@0.5**.

The competition fixes the model, so the leaderboard is climbed by fixing the **data**:

| Locked by the competition | Yours to change |
|---|---|
| YOLOv11n, official COCO `yolo11n.pt` (sha256 `0ebbc80d4a76…`) | Labels (find & fix errors) |
| `imgsz=640`, `pretrained=True` | Epochs (≤300), batch (≤128), lr, optimizer, augmentation |
| | Auxiliary training-only classes, inference confidence |

## Run on Kaggle

1. Push this folder to a GitHub repo (public, or attach it as a Kaggle Dataset instead).
2. Create a Kaggle notebook in the competition, import `notebooks/kaggle_runner.ipynb`.
3. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
4. Set `REPO_URL` in the first code cell, then **Run All**.

The notebook clones the repo and runs each script below in order, streaming output and stopping on the first failure. `submission.csv` ends up in `/kaggle/working/`.

## Scripts (run from the repo root, in order)

```
python scripts/00_check_env.py         # torch / CUDA / GPU check (fails if no GPU; --allow-cpu to override)
python scripts/01_download_kit.py      # starter kit from the 3LC CDN, sha256-verified, resumable
python scripts/02_prepare_dataset.py   # working copy: images linked, labels copied, dataset.yaml
python scripts/03_audit.py             # class balance, darkness, label sanity -> outputs/audit/
python scripts/04_train.py --name baseline            # YOLOv11n@640, CUDA + AMP, DDP over all GPUs
python scripts/05_mine_labels.py       # review queue of likely label errors + previews
python scripts/06_apply_fixes.py --apply              # apply rows you marked status=accepted
python scripts/04_train.py --name v2_fixed            # retrain on fixed labels
python scripts/07_predict.py --weights work/runs/v2_fixed/weights/best.pt
python scripts/08_validate_submission.py              # same format rules as the metric
python scripts/09_submit.py            # optional: Kaggle API upload (needs credentials)
```

`bash run_pipeline.sh 60` runs the first round (steps 0–5, 7, 8) in one go.

Paths: on Kaggle, data/runs go to `/kaggle/working/lowlight/` and outputs to `/kaggle/working/`.
Locally they go to `./work/` and `./work/outputs/`. Override with `LOWLIGHT_WORK`; point
`LOWLIGHT_KIT` at an existing starter kit to skip the download.

## The data-centric loop

1. Train a baseline (step 4).
2. `05_mine_labels.py` compares the model with the labels and writes `review/review_queue.csv`:
   - `missing_label` — confident detection with no label nearby → add it
   - `class_flip` — detection sits on a label of another class → relabel
   - `loose_box` — right class, badly fitting box → replace
   - `possible_spurious` — label the model never finds → delete
   Each candidate has a CLAHE-brightened preview (green = label, red = model).
3. Open the previews, set `status` to `accepted` or `rejected`. Don't bulk-accept: on near-dark images the model is wrong often enough that unreviewed edits cost mAP.
4. `06_apply_fixes.py --apply` (backs up labels first), retrain, compare `val_mAP50` in `provenance_<run>.json`, repeat.

On Kaggle, download `review_bundle.zip`, review, upload the edited CSV as a Dataset, set `REVIEWED_QUEUE` in the notebook and run again.

## Submission format

```
id,image_id,prediction_string
0,2015_00015,10 0.91234 0.51200 0.40100 0.21000 0.55000 4 0.40000 0.20000 0.60000 0.10000 0.08000
1,2015_00017,no box
```

715 rows, `id` = 0..714 aligned to `image_id` sorted ascending; each box is
`class conf x_center y_center width height`, YOLO-normalized to [0, 1]; `no box` when empty.
Class ids: 0 Bicycle, 1 Boat, 2 Bottle, 3 Bus, 4 Car, 5 Cat, 6 Chair, 7 Cup, 8 Dog, 9 Motorbike, 10 People, 11 Table.

## Notes

- **Prize eligibility.** The competition states that training, prediction and submission run in the 3LC Hub's Kaggle plugin, which records verified provenance for each run. This repo follows the same locked contract and writes its own `provenance_*.json`, but that is not the plugin's verified Run record. Check the Rules; if you're competing for prizes, replay your fixed dataset through the plugin's Train → Predict → Submit tabs.
- **TTA** (`07_predict.py --tta`) is off by default — confirm the Rules allow it before submitting with it.
- Hyperparameters live in `configs/train.yaml`; override any with `--set key=value`. The script refuses locked keys and out-of-bounds values.
- Ultralytics is AGPL-3.0.
