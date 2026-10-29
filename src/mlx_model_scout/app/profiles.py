"""Loading of hardware profiles and chip reference data from YAML."""

from pathlib import Path

import yaml

from mlx_model_scout.config.settings import get_settings
from mlx_model_scout.schemas.hardware import Chip, HardwareProfile


class ProfileError(ValueError):
    """Raised for unknown profiles/chips or malformed config files."""


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ProfileError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ProfileError(f"Expected a mapping at top level of {path}")
    return data


def load_chips(path: Path | None = None) -> dict[str, Chip]:
    """Load chip reference data keyed by chip id."""
    path = path or get_settings().chips_path
    raw = _read_yaml(path).get("chips", {})
    return {key: Chip(key=key, **values) for key, values in raw.items()}


def load_profiles(path: Path | None = None) -> dict[str, HardwareProfile]:
    """Load configured hardware profiles keyed by profile id."""
    path = path or get_settings().hardware_profiles_path
    raw = _read_yaml(path).get("profiles", [])
    profiles: dict[str, HardwareProfile] = {}
    for entry in raw:
        profile = HardwareProfile(**entry, source="config")
        if profile.id in profiles:
            raise ProfileError(f"Duplicate profile id: {profile.id}")
        profiles[profile.id] = profile
    return profiles


def get_profile(profile_id: str, profiles: dict[str, HardwareProfile] | None = None) -> HardwareProfile:
    """Return one profile or raise ProfileError with the list of valid ids."""
    profiles = profiles if profiles is not None else load_profiles()
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise ProfileError(
            f"Unknown profile '{profile_id}'. Available: {', '.join(sorted(profiles))}"
        ) from exc


def get_chip(chip_key: str, chips: dict[str, Chip] | None = None) -> Chip:
    """Return chip reference data or raise ProfileError."""
    chips = chips if chips is not None else load_chips()
    try:
        return chips[chip_key]
    except KeyError as exc:
        raise ProfileError(f"Unknown chip '{chip_key}'. Add it to chips.yaml.") from exc
