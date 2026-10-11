"""
Device Verification and Diagnostic Script for TensorFlow Hardware Acceleration.

Checks:
1. Operating System environment (WSL2 Linux vs Windows native).
2. NVIDIA GPU visibility via nvidia-smi.
3. TensorFlow version and CUDA build flags.
4. Physical GPU devices registered in TensorFlow.
5. GPU memory growth configuration.
6. Execution of a test tensor calculation on the active device.
"""

import os
import sys
import time
import subprocess
import platform

from src.device import configure_device


def check_nvidia_smi() -> dict:
    """Executes nvidia-smi to verify driver level GPU detection."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.strip().split("\n")
            details = [line.strip() for line in lines]
            return {"available": True, "details": details}
    except Exception:
        pass
    return {"available": False, "details": []}


def is_wsl() -> bool:
    """Checks if running inside Windows Subsystem for Linux (WSL)."""
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            version_str = f.read().lower()
            return "microsoft" in version_str or "wsl" in version_str
    except Exception:
        return False


def run_test_computation(device_string: str):
    """Executes a test matrix multiplication to verify tensor operations on target device."""
    import tensorflow as tf

    print(f"\n[Test Benchmark] Running test matrix multiplication on {device_string}...")
    with tf.device(device_string):
        size = 2000
        a = tf.random.normal([size, size], dtype=tf.float32)
        b = tf.random.normal([size, size], dtype=tf.float32)
        
        start_time = time.perf_counter()
        c = tf.matmul(a, b)
        # Force evaluation
        _ = c.numpy()
        elapsed = (time.perf_counter() - start_time) * 1000.0

    print(f" -> Computed {size}x{size} matrix product in {elapsed:.2f} ms on {device_string}")
    return elapsed


def main():
    print("=" * 70)
    print(" HANDWRITTEN DIGIT RECOGNITION - DEVICE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Environment Details
    os_name = platform.system()
    os_release = platform.release()
    wsl_active = is_wsl()

    print(f" Operating System  : {os_name} {os_release}")
    print(f" Environment Type  : {'WSL2 Linux Environment' if wsl_active else 'Native Windows / Host'}")
    print(f" Python Version    : {sys.version.split()[0]}")

    # 2. NVIDIA System Check
    smi_res = check_nvidia_smi()
    print("\n[NVIDIA Driver Check]")
    if smi_res["available"]:
        print(" -> nvidia-smi is available.")
        for d in smi_res["details"]:
            print(f"    - Hardware Detected: {d}")
    else:
        print(" -> nvidia-smi was not found or failed to return GPU details.")

    # 3. TensorFlow Configuration
    print("\n[TensorFlow Device Configuration]")
    device_info = configure_device(verbose=True)

    # 4. Verify Computation
    active_device = device_info["device_string"]
    try:
        run_test_computation(active_device)
        print("\n[Verification Result]")
        if device_info["gpu_available"]:
            print(f" [PASS] TensorFlow GPU acceleration is CONFIRMED and ACTIVE on {device_info['gpu_devices'][0].name}!")
        else:
            print(" [PASS] TensorFlow CPU execution is CONFIRMED and functional.")
            if smi_res["available"] and not wsl_active:
                print("\n[Actionable Advice for NVIDIA RTX 3050 Acceleration]:")
                print(" 1. NVIDIA RTX 3050 hardware was found by your system driver, but native Windows TensorFlow (>= 2.11) only supports CPU.")
                print(" 2. To harness GPU acceleration with your RTX 3050, run this project within WSL2 (Ubuntu).")
                print(" 3. Inside WSL2, TensorFlow directly accesses your RTX 3050 via the Windows NVIDIA driver passthrough.")
                print(" 4. Setup instructions are detailed in README.md.")
    except Exception as e:
        print(f"\n[FAIL] Computation verification failed on {active_device}: {e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
