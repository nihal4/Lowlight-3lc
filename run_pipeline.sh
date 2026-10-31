#!/usr/bin/env bash
# Run the whole first round end to end: env check -> data -> audit -> train -> mine -> predict -> validate.
# Usage: bash run_pipeline.sh [EPOCHS]
set -euo pipefail
cd "$(dirname "$0")"
EPOCHS="${1:-60}"

python scripts/00_check_env.py
python scripts/01_download_kit.py
python scripts/02_prepare_dataset.py
python scripts/03_audit.py
python scripts/04_train.py --name baseline --epochs "$EPOCHS"
python scripts/05_mine_labels.py --weights work/runs/baseline/weights/best.pt
python scripts/07_predict.py --weights work/runs/baseline/weights/best.pt
python scripts/08_validate_submission.py
echo "Done. Review outputs/review/review_queue.csv, then 06_apply_fixes.py --apply and retrain."
