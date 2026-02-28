import json
import os


def get_default_label_config_path() -> str:
    """Return the repository-local label mapping config path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "config", "labels.json")


def load_label_mapping(config_path: str | None = None) -> dict[str, int]:
    """Load and validate the allowed category-to-label mapping."""
    path = config_path or get_default_label_config_path()
    with open(path, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Label config must contain a non-empty object mapping category names to integer labels.")

    normalized: dict[str, int] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ValueError("Label config keys must be strings.")
        if not isinstance(value, int):
            raise ValueError(f"Label for category '{key}' must be an integer.")
        normalized[key] = value

    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Label config contains duplicate numeric labels.")

    return normalized
