"""Step 8 - re-check a submission.csv against the competition format (no GPU needed)."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from lowlight.common import CLASSES, banner, outputs_dir
from lowlight.submission import validate_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None, help="default: outputs/submission.csv")
    ap.add_argument("--sample", default=None, help="sample_submission.csv to compare image ids with")
    ap.add_argument("--rows", type=int, default=715)
    args = ap.parse_args()

    banner("Step 8: validate submission")
    import pandas as pd

    path = Path(args.csv) if args.csv else outputs_dir() / "submission.csv"
    df = pd.read_csv(path, keep_default_na=False)
    sample_ids = pd.read_csv(args.sample)["image_id"].astype(str).tolist() if args.sample else None
    s = validate_df(df, expected_rows=args.rows, sample_ids=sample_ids)
    s["per_class"] = dict(zip(CLASSES, s["per_class"]))
    print(json.dumps(s, indent=2))
    print(f"OK: {path} is a valid submission")


if __name__ == "__main__":
    main()
