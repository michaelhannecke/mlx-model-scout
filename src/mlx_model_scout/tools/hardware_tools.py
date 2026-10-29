"""Tool implementations: hardware detection and profiles."""

from mlx_model_scout.app.hardware import detect_hardware, detected_to_profile
from mlx_model_scout.app.profiles import load_profiles
from mlx_model_scout.db.audit import tool_audit
from mlx_model_scout.schemas.hardware import HardwareProfile


def detect_local_hardware() -> dict:
    """Detect this machine and, if possible, convert it into a profile."""
    with tool_audit("detect_hardware", {}) as audit:
        hw = detect_hardware()
        profile = detected_to_profile(hw)
        audit["summary"] = hw.cpu_brand or hw.note
        return {
            "detected": hw.model_dump(),
            "profile": profile.model_dump() if profile else None,
        }


def list_hardware_profiles() -> list[HardwareProfile]:
    """Return configured profiles plus the detected local machine (if any)."""
    profiles = list(load_profiles().values())
    local = detected_to_profile(detect_hardware())
    if local:
        profiles.insert(0, local)
    return profiles


def resolve_profile(profile_id: str) -> HardwareProfile:
    """Resolve 'local' to the detected machine, otherwise look up config."""
    if profile_id == "local":
        local = detected_to_profile(detect_hardware())
        if local is None:
            raise ValueError("Local hardware not detectable here (not macOS / inside a container).")
        return local
    from mlx_model_scout.app.profiles import get_profile

    return get_profile(profile_id)
