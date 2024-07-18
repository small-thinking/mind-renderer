"""Run the test with the command:
    poetry run pytest tests/utils/test_config_loader.py
"""

from unittest.mock import mock_open, patch

import pytest
import yaml

from mind_renderer.utils.config_loader import ConfigLoader


class TestConfigLoader:
    @pytest.fixture
    def sample_config(self):
        return {
            "story": "写一个儿童故事，用于正面影响小朋友注意卫生。并且喜欢吃蔬菜。",
            "num_sections": 2,
            "section_length": 1000,
            "genres": "fantasy,scifi",
            "writing_style": "third person",
            "gen_thumbnail": True,
            "text_models": {"provider": "DeepSeek", "lm_name": "deep-seek-chat"},
        }

    @patch("os.path.exists")
    def test_load_config_success(self, mock_exists, sample_config):
        mock_exists.return_value = True
        mock_config = yaml.dump(sample_config)
        with patch("builtins.open", mock_open(read_data=mock_config)):
            config_loader = ConfigLoader("mock_config.yaml")
            assert config_loader.get_config() == sample_config

    def test_load_config_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ConfigLoader("non_existent_config.yaml")

    @patch("os.path.exists")
    def test_get_text_model_config(self, mock_exists, sample_config):
        mock_exists.return_value = True
        mock_config = yaml.dump(sample_config)
        with patch("builtins.open", mock_open(read_data=mock_config)):
            config_loader = ConfigLoader("mock_config.yaml")
            assert config_loader.get_text_model_config() == {"provider": "DeepSeek", "lm_name": "deep-seek-chat"}

    @patch("os.path.exists")
    def test_get_text_model_config_missing(self, mock_exists):
        mock_exists.return_value = True
        config_without_text_models = {"story": "Test story"}
        mock_config = yaml.dump(config_without_text_models)
        with patch("builtins.open", mock_open(read_data=mock_config)):
            config_loader = ConfigLoader("mock_config.yaml")
            assert config_loader.get_text_model_config() == {}
