"""Local hardware detection (macOS / Apple Silicon only).

Inside a Linux container (Apple `container`, Docker, Podman) the host hardware
is NOT visible: the container runs in a Linux VM. In that case detection
returns `is_macos=False` and callers must fall back to configured profiles.
"""

import json
import platform
import subprocess
from pathlib import Path

from mlx_model_scout.app.profiles import load_chips
from mlx_model_scout.schemas.hardware import DetectedHardware, HardwareProfile

GIB = 1024**3


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True)
    except OSError, subprocess.SubprocessError:
        return None
    value = out.stdout.strip()
    return value or None


def _gpu_cores() -> int | None:
    """GPU core count via `system_profiler` (the CPU brand string does not contain it)."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        gpus = json.loads(out.stdout).get("SPDisplaysDataType", [])
    except OSError, subprocess.SubprocessError, ValueError:
        return None
    for gpu in gpus:
        cores = str(gpu.get("sppci_cores", ""))
        if cores.isdigit():
            return int(cores)
    return None


def _in_container() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def detect_hardware() -> DetectedHardware:
    """Detect the local machine. Never raises; returns a descriptive result."""
    if platform.system() != "Darwin":
        return DetectedHardware(
            is_macos=False,
            is_apple_silicon=False,
            running_in_container=_in_container(),
            note="Not running on macOS (e.g. inside a container). Use a configured profile.",
        )

    brand = _sysctl("machdep.cpu.brand_string")
    memsize = _sysctl("hw.memsize")
    wired_mb = _sysctl("iogpu.wired_limit_mb")

    memory_gb = round(int(memsize) / GIB, 1) if memsize and memsize.isdigit() else None
    wired_gb = None
    if wired_mb and wired_mb.isdigit() and int(wired_mb) > 0:
        wired_gb = round(int(wired_mb) / 1024, 1)

    return DetectedHardware(
        is_macos=True,
        is_apple_silicon=platform.machine() == "arm64",
        cpu_brand=brand,
        gpu_cores=_gpu_cores(),
        unified_memory_gb=memory_gb,
        wired_limit_gb=wired_gb,
        note=None if wired_gb else "iogpu.wired_limit_mb not set; macOS default limit applies.",
    )


# Fallback when the GPU core count is unknown: the largest known variant per chip.
_LARGEST_VARIANT = {
    ("m1", "ultra"): "m1-ultra-64c",
    ("m2", "ultra"): "m2-ultra-76c",
    ("m3", "ultra"): "m3-ultra-80c",
    ("m1", "max"): "m1-max-32c",
    ("m2", "max"): "m2-max-38c",
    ("m3", "max"): "m3-max-40c",
    ("m4", "max"): "m4-max-40c",
    ("m5", "max"): "m5-max-40c",
    ("m4", "pro"): "m4-pro-20c",
    ("m5", "pro"): "m5-pro-20c",
}


def chip_key_from_brand(brand: str | None, gpu_cores: int | None = None) -> str | None:
    """Map a CPU brand string (+ GPU core count, if known) to a chips.yaml key.

    With `gpu_cores` the exact key (e.g. `m5-pro-20c`) is returned; without it,
    the largest known variant, and the caller should flag the assumption.
    The returned key may not exist in chips.yaml; callers must check.
    """
    if not brand:
        return None
    b = brand.lower()
    gen = next((g for g in ("m5", "m4", "m3", "m2", "m1") if g in b), None)
    if gen is None:
        return None
    tier = next((t for t in ("ultra", "max", "pro") if t in b), None)
    if gpu_cores:
        return f"{gen}-{tier}-{gpu_cores}c" if tier else f"{gen}-{gpu_cores}c"
    return _LARGEST_VARIANT.get((gen, tier)) if tier else None


def detected_to_profile(hw: DetectedHardware, known_chips: set[str] | None = None) -> HardwareProfile | None:
    """Convert a detection result into a usable profile, if possible."""
    if not (hw.is_macos and hw.is_apple_silicon and hw.unified_memory_gb):
        return None
    known = known_chips if known_chips is not None else set(load_chips())
    notes = None
    chip = chip_key_from_brand(hw.cpu_brand, hw.gpu_cores)
    if chip not in known:
        fallback = chip_key_from_brand(hw.cpu_brand)
        chip = fallback if fallback in known else "unknown"
        if chip != "unknown":
            notes = f"GPU core count not matched; assumed largest variant ({chip})."
    return HardwareProfile(
        id="local",
        name=f"This Mac ({hw.cpu_brand or 'Apple Silicon'})",
        chip=chip,
        unified_memory_gb=hw.unified_memory_gb,
        wired_limit_gb=hw.wired_limit_gb,
        notes=notes,
        source="detected",
    )
