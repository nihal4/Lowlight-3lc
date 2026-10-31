"""Step 0 - check Python, PyTorch, CUDA and GPUs before spending time on anything else."""
import argparse
import platform
import shutil
import subprocess

import _bootstrap  # noqa: F401
from lowlight.common import ON_KAGGLE, banner, work_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-cpu", action="store_true", help="don't fail when no CUDA GPU is found")
    args = ap.parse_args()

    banner("Step 0: environment check")
    print(f"Python      : {platform.python_version()}")
    print(f"On Kaggle   : {ON_KAGGLE}")
    print(f"Work dir    : {work_dir()}")

    import torch

    print(f"PyTorch     : {torch.__version__} (built for CUDA {torch.version.cuda})")
    try:
        import ultralytics

        print(f"Ultralytics : {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics : NOT INSTALLED (pip install -r requirements.txt)")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"CUDA        : available, {n} GPU(s)")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i}  {p.name}  {p.total_memory / 1e9:.1f} GB  (sm_{p.major}{p.minor})")
        torch.backends.cudnn.benchmark = True
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        _ = (x @ x).sum().item()
        print("  fp16 matmul on GPU: OK")
        if n > 1:
            print("  -> training will use DDP across all GPUs (device=auto)")
    else:
        msg = ("No CUDA GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2 (or P100), "
               "then restart the session.")
        if not args.allow_cpu:
            raise SystemExit(msg)
        print("WARNING:", msg)

    if shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
