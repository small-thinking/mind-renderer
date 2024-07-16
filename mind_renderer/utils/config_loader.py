"""Config loader module for loading configuration from a YAML file.
"""

import os
from typing import Any, Dict

import yaml


class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as config_file:
            return yaml.safe_load(config_file)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_text_model_config(self) -> Dict[str, str]:
        return self.config.get("text_models", {})
