"""Step 9 (optional) - upload submission.csv to Kaggle with the Kaggle API.

Needs Kaggle credentials, one of:
  * KAGGLE_API_TOKEN env var (new KGAT_... token), or
  * KAGGLE_USERNAME + KAGGLE_KEY env vars, or
  * ~/.kaggle/kaggle.json / ~/.kaggle/access_token
In a Kaggle notebook, store them as Add-ons -> Secrets; the runner notebook
exports them. Without credentials this step just tells you to upload by hand.
The competition allows a limited number of submissions per day.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from lowlight.common import COMPETITION_SLUG, banner, outputs_dir


def have_credentials() -> bool:
    home = Path.home() / ".kaggle"
    return bool(os.environ.get("KAGGLE_API_TOKEN")
                or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
                or (home / "kaggle.json").exists() or (home / "access_token").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None)
    ap.add_argument("-m", "--message", default="yolo11n@640 lowlight pipeline")
    args = ap.parse_args()

    banner("Step 9: submit to Kaggle")
    path = Path(args.csv) if args.csv else outputs_dir() / "submission.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    if not have_credentials():
        print("No Kaggle credentials found - skipping upload.")
        print(f"Upload {path} by hand on the competition's Submit page.")
        return
    if not shutil.which("kaggle"):
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION_SLUG, "-f", str(path), "-m", args.message]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode:
        raise SystemExit("Submission failed (have you joined the competition and accepted the rules?)")
    subprocess.run(["kaggle", "competitions", "submissions", "-c", COMPETITION_SLUG], check=False)


if __name__ == "__main__":
    main()
