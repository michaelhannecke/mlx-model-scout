"""Hardware-related Pydantic models."""

from typing import Literal

from pydantic import BaseModel, Field


class Chip(BaseModel):
    """Reference data for an Apple Silicon chip variant."""

    key: str
    display_name: str
    memory_bandwidth_gbs: float | None = Field(
        default=None, description="Peak unified-memory bandwidth in GB/s; None if unknown."
    )
    verified: bool = False


class HardwareProfile(BaseModel):
    """A real (detected) or hypothetical (configured) machine."""

    id: str
    name: str
    chip: str = Field(description="Key into chips.yaml")
    unified_memory_gb: float = Field(gt=0)
    wired_limit_gb: float | None = Field(
        default=None, description="GPU wired-memory limit per node; None = use default fraction."
    )
    reserved_gb: float = Field(default=0.0, ge=0, description="Memory kept free per node.")
    nodes: int = Field(default=1, ge=1)
    notes: str | None = None
    source: Literal["config", "detected"] = "config"


class DetectedHardware(BaseModel):
    """Raw result of local hardware detection (macOS only)."""

    is_macos: bool
    is_apple_silicon: bool
    cpu_brand: str | None = None
    gpu_cores: int | None = Field(default=None, description="GPU core count from system_profiler.")
    unified_memory_gb: float | None = None
    wired_limit_gb: float | None = Field(
        default=None, description="Explicit iogpu.wired_limit_mb if set (0 = macOS default)."
    )
    running_in_container: bool = False
    note: str | None = None
