"""Automated environment setup and training launcher for Apple Silicon healthcare ML."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python3"
PYTHON_310_CMD = "python3.10"
MAX_TF_SUPPORTED = (3, 11)

REQUIRED_PACKAGES = [
    "tensorflow-macos",
    "tensorflow-metal",
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "xgboost",
]


def _run_cmd(command: list[str], description: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with consistent logging and error handling."""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=check,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout.strip())
        if exc.stderr:
            print(exc.stderr.strip())
        raise RuntimeError(f"Command failed ({description}) with exit code {exc.returncode}.") from exc

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

    return result


def _warn_if_host_python_unsupported() -> None:
    """Warn when current interpreter is newer than TensorFlow-supported versions."""
    if (sys.version_info.major, sys.version_info.minor) > MAX_TF_SUPPORTED:
        print("[WARN] TensorFlow does not support Python 3.13. Switching to Python 3.10 is required.")


def _get_python_version(python_exec: str) -> tuple[int, int]:
    """Return (major, minor) for the given Python executable."""
    result = _run_cmd(
        [python_exec, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        description=f"Checking Python version for {python_exec}",
        check=True,
    )
    raw = result.stdout.strip()
    major_str, minor_str = raw.split(".")[:2]
    return int(major_str), int(minor_str)


def _ensure_supported_venv_python() -> None:
    """Ensure venv Python is TensorFlow-compatible before installing dependencies."""
    if not VENV_PYTHON.exists():
        raise RuntimeError("Virtual environment Python not found at venv/bin/python3.")

    major, minor = _get_python_version(str(VENV_PYTHON))
    if (major, minor) > MAX_TF_SUPPORTED:
        raise RuntimeError(
            "TensorFlow install skipped: venv Python is unsupported. "
            "Please recreate venv with Python 3.10."
        )


def _ensure_venv() -> None:
    """Create virtual environment if it does not already exist."""
    python_310 = shutil.which(PYTHON_310_CMD)
    if python_310 is None:
        print("[ERROR] Please install Python 3.10 using: brew install python@3.10")
        raise RuntimeError("Python 3.10 not found on PATH.")

    if VENV_PYTHON.exists():
        major, minor = _get_python_version(str(VENV_PYTHON))
        if (major, minor) <= MAX_TF_SUPPORTED:
            print("[INFO] Virtual environment already exists at ./venv")
            return

        print("[WARN] Existing venv uses unsupported Python version. Recreating with Python 3.10...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    print("[INFO] Creating virtual environment...")
    _run_cmd(
        [python_310, "-m", "venv", "venv"],
        description="Creating virtual environment (python3.10 -m venv venv)",
    )
    print("[INFO] Virtual environment created successfully.")


def _upgrade_pip_and_install_packages() -> None:
    """Upgrade pip and install required project dependencies."""
    _ensure_supported_venv_python()
    print("[INFO] Installing dependencies...")
    _run_cmd([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")

    install_cmd = [str(VENV_PYTHON), "-m", "pip", "install", *REQUIRED_PACKAGES]
    _run_cmd(install_cmd, "Installing required dependencies")


def _verify_tensorflow() -> tuple[bool, str, str]:
    """Verify TensorFlow import and detect active compute device."""
    print("[INFO] Verifying TensorFlow...")
    verify_code = (
        "import os\n"
        "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'\n"
        "import tensorflow as tf\n"
        "print(tf.__version__)\n"
        "devices = tf.config.list_physical_devices()\n"
        "print('Using GPU (Metal)' if tf.config.list_physical_devices('GPU') else 'Using CPU')\n"
        "for d in devices:\n"
        "    print(f'{d.device_type}:{d.name}')\n"
    )

    result = _run_cmd(
        [str(VENV_PYTHON), "-c", verify_code],
        description="Verifying TensorFlow installation and available devices",
        check=False,
    )

    if result.returncode != 0:
        print("[ERROR] TensorFlow import failed.")
        print("[ERROR] Please check compatible versions for tensorflow-macos/tensorflow-metal.")
        return False, "unknown", "unknown"

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    tf_version = lines[0] if lines else "unknown"
    active_device = lines[1] if len(lines) > 1 else "unknown"

    print(f"[INFO] TensorFlow version: {tf_version}")
    print(f"[INFO] Device being used: {active_device}")
    if len(lines) > 2:
        print("[INFO] Available devices:")
        for line in lines[2:]:
            print(f"  - {line}")

    return True, tf_version, active_device


def _run_training_pipeline() -> None:
    """Run the centralized training pipeline in the configured virtual environment."""
    if not VENV_PYTHON.exists():
        raise RuntimeError("Virtual environment Python not found at venv/bin/python3.")

    print("\n[INFO] Starting training pipeline...")

    _run_cmd(
        [str(VENV_PYTHON), "src/pipelines/train_eval_pipeline.py"],
        description="Running training pipeline",
        check=True,
    )


def main() -> None:
    """Set up environment, verify TensorFlow, then launch training pipeline."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    print("[INFO] Healthcare ML setup started.")

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("[INFO] Apple Silicon detected (M-series). Installing optimized TensorFlow packages.")
    else:
        print("[WARN] Non-Apple-Silicon environment detected; continuing with requested package set.")

    try:
        _warn_if_host_python_unsupported()
        _ensure_venv()
        _upgrade_pip_and_install_packages()

        tf_ok, _, _ = _verify_tensorflow()
        if not tf_ok:
            sys.exit(1)

        print("\n[INFO] Setup successful")
        print("[INFO] Training started")
        _run_training_pipeline()
        print("[INFO] Training completed")

        print("\n[SUCCESS] Setup complete and training command executed successfully.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n[ERROR] Setup failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
