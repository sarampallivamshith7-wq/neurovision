"""Configuration loader for NeuroVision with dot-attribute dictionary access."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


class ConfigDict(dict):
    """A dictionary subclass supporting attribute-style access (e.g. cfg.training.batch_size)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = ConfigDict(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to standard Python dict."""
        out = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


def load_config(config_path: Union[str, Path] = "configs/phase1.yaml") -> ConfigDict:
    """Load a YAML configuration file into a ConfigDict."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    return ConfigDict(raw_data or {})
