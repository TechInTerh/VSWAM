"""
PyTorch + NVIDIA GPU installation sanity check.

Run with:
    python check_pytorch_gpu.py

Exit codes:
    0 = PyTorch OK (CPU or CUDA)
    1 = PyTorch not installed / failed to import
    2 = PyTorch OK but CUDA/GPU issues detected
"""

import shutil
import subprocess
import sys

SEP = "=" * 60


def check_pytorch():
    """Check that PyTorch imports and report its version."""
    print(SEP)
    print("1. Checking PyTorch installation")
    print(SEP)
    try:
        import torch
    except ImportError as exc:
        print(f"[FAIL] Could not import PyTorch: {exc}")
        print("       Install it with: pip install torch")
        sys.exit(1)

    print("[OK]   PyTorch is installed")
    print(f"       Version           : {torch.__version__}")
    print(f"       CUDA compiled with: {torch.version.cuda}")
    print(f"       cuDNN version     : {torch.backends.cudnn.version()}")
    return torch


def check_nvidia_drivers():
    """Check that the NVIDIA driver is present via nvidia-smi."""
    print()
    print(SEP)
    print("2. Checking NVIDIA drivers (nvidia-smi)")
    print(SEP)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("[FAIL] nvidia-smi not found on PATH.")
        print("       The NVIDIA driver is probably not installed,")
        print("       or this machine has no NVIDIA GPU.")
        return False

    try:
        result = subprocess.run(
            [nvidia_smi],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("[FAIL] nvidia-smi timed out.")
        return False

    if result.returncode != 0:
        print(f"[FAIL] nvidia-smi exited with code {result.returncode}.")
        print(result.stderr.strip())
        return False

    print("[OK]   NVIDIA driver detected via nvidia-smi:")
    print()
    print(result.stdout)
    return True


def check_cuda(torch):
    """Check CUDA availability from PyTorch's point of view."""
    print(SEP)
    print("3. Checking CUDA availability in PyTorch")
    print(SEP)
    if not torch.cuda.is_available():
        print("[WARN] torch.cuda.is_available() returned False.")
        print("       Possible causes:")
        print("       - CPU-only PyTorch build (reinstall with CUDA, e.g.")
        print("         pip install torch --index-url https://download.pytorch.org/whl/cu124)")
        print("       - NVIDIA driver missing or too old for this CUDA version")
        print("       - No NVIDIA GPU in this machine")
        return False

    print(f"[OK]   CUDA is available: {torch.version.cuda}")
    print(f"       Device count: {torch.cuda.device_count()}")
    print(f"       Current device: {torch.cuda.current_device()}")
    print()
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"       GPU {i}: {props.name}")
        print(f"         - Compute capability : {props.major}.{props.minor}")
        print(f"         - Total memory       : {props.total_memory / 1024**3:.1f} GB")
        print(f"         - Multiprocessors   : {props.multi_processor_count}")
    return True


def run_gpu_stress_test(torch):
    """Run a tiny tensor computation on the GPU to confirm it actually works."""
    print()
    print(SEP)
    print("4. Running a small GPU computation test")
    print(SEP)
    if not torch.cuda.is_available():
        print("[SKIP] No CUDA available, skipping GPU test.")
        return False

    try:
        device = torch.device("cuda")
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        torch.cuda.synchronize()  # make sure kernels actually ran
        c = (a @ b).sum().item()
        print(f"[OK]   Matrix multiplication on GPU succeeded (result: {c:.3e})")

        # Simple sanity check on GPU vs CPU
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        if (x * 2).cpu().tolist() != [2.0, 4.0, 6.0]:
            print("[WARN] GPU computation returned an unexpected result.")
            return False
        print("[OK]   GPU/CPU transfer sanity check passed.")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] GPU computation raised an exception: {exc}")
        return False


def main():
    print()
    torch = check_pytorch()
    drivers_ok = check_nvidia_drivers()
    cuda_ok = check_cuda(torch)
    compute_ok = run_gpu_stress_test(torch)

    print()
    print(SEP)
    print("Summary")
    print(SEP)
    print("PyTorch installed        : OK")
    print(f"NVIDIA drivers installed : {'OK' if drivers_ok else 'NOT FOUND'}")
    print(f"CUDA available to PyTorch : {'YES' if cuda_ok else 'NO'}")
    print(f"GPU computation test      : {'PASSED' if compute_ok else 'SKIPPED/FAILED'}")

    if cuda_ok and compute_ok:
        print("\nAll checks passed. PyTorch is ready to use the GPU.\n")
        sys.exit(0)
    elif drivers_ok and not cuda_ok:
        print("\nDrivers look fine but PyTorch cannot see CUDA.")
        print("Reinstall PyTorch with a CUDA-enabled build.\n")
        sys.exit(2)
    else:
        print("\nPyTorch works, but GPU support is not available.")
        print("You can still run on CPU.\n")
        sys.exit(2)


if __name__ == "__main__":
    main()