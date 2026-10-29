"""Resources `hardware://profiles` and `hardware://chips`."""

from mlx_model_scout.app.profiles import load_chips
from mlx_model_scout.tools.hardware_tools import list_hardware_profiles


def hardware_profiles() -> list[dict]:
    """All hardware profiles: configured (YAML) plus the detected 'local' Mac, if any."""
    return [p.model_dump() for p in list_hardware_profiles()]


def chips() -> dict:
    """Apple Silicon reference data (memory bandwidth per chip) keyed by chip id."""
    return {k: c.model_dump() for k, c in load_chips().items()}
