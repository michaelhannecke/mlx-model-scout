from mlx_model_scout.app.hardware import chip_key_from_brand, detected_to_profile
from mlx_model_scout.schemas.hardware import DetectedHardware

KNOWN = {"m5-pro-20c", "m5-max-32c", "m5-max-40c", "m4-pro-20c"}


def _hw(brand: str, cores: int | None) -> DetectedHardware:
    return DetectedHardware(
        is_macos=True, is_apple_silicon=True, cpu_brand=brand, gpu_cores=cores, unified_memory_gb=64.0
    )


def test_chip_key_uses_gpu_cores():
    assert chip_key_from_brand("Apple M5 Max", 32) == "m5-max-32c"
    assert chip_key_from_brand("Apple M5 Pro", 20) == "m5-pro-20c"
    assert chip_key_from_brand("Apple M5", 10) == "m5-10c"


def test_chip_key_without_cores_falls_back_to_largest_variant():
    assert chip_key_from_brand("Apple M5 Max") == "m5-max-40c"
    assert chip_key_from_brand("Apple M5 Pro") == "m5-pro-20c"
    assert chip_key_from_brand("Apple M5") is None
    assert chip_key_from_brand("Intel Core i9") is None


def test_detected_profile_exact_match():
    profile = detected_to_profile(_hw("Apple M5 Max", 32), KNOWN)
    assert profile.chip == "m5-max-32c"
    assert profile.notes is None


def test_detected_profile_unknown_core_count_is_flagged():
    profile = detected_to_profile(_hw("Apple M5 Max", 36), KNOWN)
    assert profile.chip == "m5-max-40c"
    assert "assumed largest variant" in profile.notes


def test_detected_profile_unmapped_chip_is_unknown():
    assert detected_to_profile(_hw("Apple M5", 10), KNOWN).chip == "unknown"
