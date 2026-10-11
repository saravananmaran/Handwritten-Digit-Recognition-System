"""
Device configuration and GPU acceleration utilities for TensorFlow.

Handles:
1. Detection of physical GPU devices.
2. Dynamic GPU memory growth configuration (crucial for RTX 3050 4GB VRAM).
3. Graceful CPU fallback with clear status logging when no GPU is detected.
4. Unified device reporting across training, inference, and Streamlit UI.
"""

import os
import sys
from typing import Dict, Any


def configure_device(verbose: bool = True) -> Dict[str, Any]:
    """
    Detects available hardware devices, configures GPU memory growth if supported,
    and returns device configuration status.

    Returns:
        dict:
            - 'gpu_available': bool (True ONLY if TensorFlow actually detects a GPU)
            - 'device_name': str ('GPU' or 'CPU')
            - 'device_string': str ('/device:GPU:0' or '/device:CPU:0')
            - 'gpu_devices': list of detected GPU physical devices
            - 'gpu_count': int
            - 'memory_growth_enabled': bool
            - 'tensorflow_version': str
            - 'built_with_cuda': bool
            - 'status_message': str
    """
    import tensorflow as tf

    tf_version = tf.__version__
    cuda_built = tf.test.is_built_with_cuda()
    physical_gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(physical_gpus) > 0
    memory_growth_enabled = False

    if gpu_available:
        device_name = "GPU"
        device_string = "/device:GPU:0"
        # Configure memory growth on all detected GPUs
        # This prevents TensorFlow from hogging all 4GB VRAM on RTX 3050 immediately
        growth_success_count = 0
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                growth_success_count += 1
            except (RuntimeError, ValueError) as e:
                if verbose:
                    print(f"[Device Setup] Warning: Could not set memory growth on {gpu}: {e}")
        memory_growth_enabled = (growth_success_count == len(physical_gpus))

        gpu_names = [getattr(g, 'name', str(g)) for g in physical_gpus]
        status_message = (
            f"GPU Acceleration ACTIVE: Detected {len(physical_gpus)} GPU device(s) "
            f"({', '.join(gpu_names)}). Dynamic memory growth: {memory_growth_enabled}."
        )
    else:
        device_name = "CPU"
        device_string = "/device:CPU:0"
        # Determine helpful diagnostic hint
        is_windows_native = sys.platform == "win32" and "microsoft" not in os.uname().release.lower() if hasattr(os, "uname") else sys.platform == "win32"
        if is_windows_native:
            reason = (
                "Native Windows TensorFlow (>= 2.11) does not support direct GPU access. "
                "To accelerate on NVIDIA RTX 3050, run this project within WSL2 (Windows Subsystem for Linux)."
            )
        else:
            reason = (
                "No GPU was registered by TensorFlow. Ensure NVIDIA drivers, CUDA toolkit, "
                "and cuDNN libraries are properly installed in WSL2."
            )

        status_message = (
            f"CPU Fallback Mode: No TensorFlow GPU detected. Executing on CPU. ({reason})"
        )

    if verbose:
        print("=" * 65)
        print(" TENSORFLOW HARDWARE ACCELERATION STATUS")
        print("=" * 65)
        print(f" TensorFlow Version  : {tf_version}")
        print(f" Built with CUDA     : {cuda_built}")
        print(f" Detected GPU Devices: {len(physical_gpus)}")
        if gpu_available:
            for idx, g in enumerate(physical_gpus):
                print(f"   -> GPU {idx}: {g.name}")
            print(f" Memory Growth       : {'Enabled' if memory_growth_enabled else 'Default'}")
            print(f" Execution Target    : GPU (Hardware Accelerated)")
        else:
            print(" Execution Target    : CPU (Fallback Mode)")
            print(f" Note                : {status_message}")
        print("=" * 65)

    return {
        "gpu_available": gpu_available,
        "device_name": device_name,
        "device_string": device_string,
        "gpu_devices": physical_gpus,
        "gpu_count": len(physical_gpus),
        "memory_growth_enabled": memory_growth_enabled,
        "tensorflow_version": tf_version,
        "built_with_cuda": cuda_built,
        "status_message": status_message
    }


def get_device_summary() -> Dict[str, Any]:
    """
    Non-verbose, lightweight query for UI and diagnostics.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return {
            "gpu_available": len(gpus) > 0,
            "device_name": "GPU" if len(gpus) > 0 else "CPU",
            "gpu_count": len(gpus),
            "gpu_devices": [g.name for g in gpus],
            "tensorflow_version": tf.__version__,
            "is_wsl": "microsoft" in getattr(os, "uname", lambda: type("U", (), {"release": ""})())().release.lower() if hasattr(os, "uname") else False
        }
    except Exception as e:
        return {
            "gpu_available": False,
            "device_name": "CPU",
            "gpu_count": 0,
            "gpu_devices": [],
            "tensorflow_version": "N/A",
            "error": str(e)
        }
