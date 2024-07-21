"""Config loader module for loading configuration from a YAML file.
"""

import os
from typing import Any, Dict

import yaml


class ConfigLoader:
    def __init__(self, config_path=None):
        config_path = config_path or "./story_gen_config.yaml"
        self.config = self._load_config(config_path=config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_text_model_config(self) -> Dict[str, str]:
        return self.config.get("text_models", {})

    def get_value(self, key: str, default=None) -> str:
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value or default
