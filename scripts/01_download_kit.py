"""Step 1 - download the competition starter kit (~625 MB).

The images are NOT on the Kaggle Data tab (that only has sample_submission.csv).
They come from 3LC's CDN, the same source the official 3LC Hub plugin uses.
Every shard and every extracted file is checked against the published sha256
manifest. Re-running resumes partial downloads and skips finished work.

Already have the kit (e.g. attached as a Kaggle Dataset)? Set LOWLIGHT_KIT to
the folder holding its dataset.yaml and skip this step, or pass --kit-dir.
"""
import argparse
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from lowlight.common import KIT_PREFIX, KIT_VERSION, banner, kit_dir, sha256_file, work_dir

CHUNK = 1 << 20


def fetch(url: str, dest: Path, expected_bytes: int, expected_sha: str, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size == expected_bytes and sha256_file(dest) == expected_sha:
        print(f"  [skip] {dest.name} already verified")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                mode = "ab" if (start and r.status == 206) else "wb"
                done = start if mode == "ab" else 0
                t0, last = time.time(), 0.0
                with open(part, mode) as f:
                    while True:
                        block = r.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        if time.time() - last > 5:
                            last = time.time()
                            mbps = (done - start) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"    {dest.name}: {done / 1e6:6.1f}/{expected_bytes / 1e6:.1f} MB  ({mbps:.1f} MB/s)", flush=True)
            digest = sha256_file(part)
            if digest != expected_sha:
                part.unlink(missing_ok=True)
                raise IOError(f"sha256 mismatch for {dest.name}")
            part.replace(dest)
            print(f"  [ok]   {dest.name} ({expected_bytes / 1e6:.1f} MB)")
            return
        except Exception as exc:  # network blips: retry, resume from .part
            print(f"  attempt {attempt}/{retries} failed for {dest.name}: {exc}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def safe_extract(zip_path: Path, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"unsafe path in archive: {member.filename}")
        z.extractall(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-dir", default=None, help="use an existing kit folder (contains dataset.yaml)")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--full-verify", action="store_true", help="sha256 every extracted file (slower)")
    args = ap.parse_args()

    banner("Step 1: starter kit")
    if args.kit_dir:
        k = Path(args.kit_dir)
        if not (k / "dataset.yaml").exists():
            raise SystemExit(f"No dataset.yaml in {k}")
        print(f"Using existing kit: {k}\nExport LOWLIGHT_KIT={k} for the next steps.")
        return

    target = kit_dir()
    version_dir = work_dir() / "data" / KIT_VERSION
    version_dir.mkdir(parents=True, exist_ok=True)
    done_flag = version_dir / ".complete"
    if done_flag.exists() and (target / "dataset.yaml").exists():
        print(f"Kit already downloaded and verified: {target}")
        return

    print(f"Source: {KIT_PREFIX}")
    manifest_path = version_dir / "manifest.json"
    with urllib.request.urlopen(f"{KIT_PREFIX}/manifest.json", timeout=60) as r:
        manifest_path.write_bytes(r.read())
    manifest = json.loads(manifest_path.read_text())
    print(f"Manifest: {manifest['file_count']} files, {manifest['total_bytes'] / 1e6:.0f} MB, "
          f"{len(manifest['archives'])} shards")

    shards = version_dir / "_shards"
    shards.mkdir(exist_ok=True)
    for a in manifest["archives"]:
        fetch(f"{KIT_PREFIX}/{a['name']}", shards / a["name"], a["bytes"], a["sha256"])

    print("Extracting ...")
    for a in manifest["archives"]:
        safe_extract(shards / a["name"], version_dir)

    print("Verifying extracted files ...")
    bad = []
    for f in manifest["files"]:
        p = version_dir / f["path"]
        if not p.exists() or p.stat().st_size != f["bytes"]:
            bad.append(f["path"])
        elif args.full_verify and sha256_file(p) != f["sha256"]:
            bad.append(f["path"])
    if bad:
        raise SystemExit(f"{len(bad)} files failed verification, e.g. {bad[:3]}. Re-run this step.")

    if not args.keep_archives:
        shutil.rmtree(shards, ignore_errors=True)
    done_flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for split in ("train", "val", "test"):
        n = len(list((target / "images" / split).glob("*")))
        print(f"  images/{split}: {n}")
    print(f"Kit ready: {target}")


if __name__ == "__main__":
    main()
