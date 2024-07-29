"""Run the test with command:
    poetry run pytest tests/utils/test_custom_lm.py
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from mind_renderer.utils.custom_lm import DeepSeek
from mind_renderer.utils.logger import Logger


class TestDeepSeek:
    @pytest.fixture
    def deepseek_instance(self):
        logger = Logger(__name__)
        return DeepSeek(model="deepseek-chat", logger=logger, api_key="test_api_key")

    def test_init(self, deepseek_instance):
        assert deepseek_instance.provider == "openai"
        assert deepseek_instance.api_base == "https://api.deepseek.com"
        assert deepseek_instance.api_key == "test_api_key"
        assert deepseek_instance.model_type == "chat"

    def test_init_without_api_key(self):
        logger = Logger(__name__)
        with pytest.raises(ValueError, match="API key must be provided"):
            DeepSeek(model="deepseek-chat", logger=logger)

    @patch("requests.post")
    def test_basic_request_chat(self, mock_post, deepseek_instance):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_post.return_value = mock_response

        response = deepseek_instance.basic_request("Test prompt")
        assert response == {"choices": [{"message": {"content": "Test response"}}]}
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_basic_request_text(self, mock_post):
        logger = Logger(__name__)
        deepseek_instance = DeepSeek(model="deepseek-text", logger=logger, api_key="test_api_key", model_type="text")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"text": "Test response"}]}
        mock_post.return_value = mock_response

        response = deepseek_instance.basic_request("Test prompt")
        assert response == {"choices": [{"text": "Test response"}]}
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_request_with_backoff(self, mock_post, deepseek_instance):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
        mock_post.side_effect = [requests.exceptions.RequestException("API error"), mock_response]

        response = deepseek_instance.request("Test prompt")
        assert response == {"choices": [{"message": {"content": "Test response"}}]}
        assert mock_post.call_count == 2

    def test_get_choice_text_chat(self, deepseek_instance):
        choice = {"message": {"content": "Test content"}}
        assert deepseek_instance._get_choice_text(choice) == "Test content"

    def test_get_choice_text_text(self):
        logger = Logger(__name__)
        deepseek_instance = DeepSeek(model="deepseek-text", logger=logger, api_key="test_api_key", model_type="text")
        choice = {"text": "Test content"}
        assert deepseek_instance._get_choice_text(choice) == "Test content"

    @patch.object(DeepSeek, "request")
    def test_call_method(self, mock_request, deepseek_instance):
        mock_request.return_value = {
            "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }

        result = deepseek_instance("Test prompt")
        assert result == ["Test response"]
        mock_request.assert_called_once_with("Test prompt")

    @patch.object(DeepSeek, "request")
    def test_call_method_with_logprobs(self, mock_request, deepseek_instance):
        mock_request.return_value = {
            "choices": [
                {
                    "message": {"content": "Test response"},
                    "finish_reason": "stop",
                    "logprobs": {"tokens": ["Test"], "token_logprobs": [-1.0]},
                }
            ],
            "usage": {"total_tokens": 10},
        }

        result = deepseek_instance("Test prompt", logprobs=True)
        assert result == [{"text": "Test response", "logprobs": {"tokens": ["Test"], "token_logprobs": [-1.0]}}]

    @patch.object(DeepSeek, "request")
    def test_call_method_incomplete_response(self, mock_request, deepseek_instance):
        mock_request.return_value = {
            "choices": [{"message": {"content": "Incomplete"}, "finish_reason": "length"}],
            "usage": {"total_tokens": 10},
        }

        result = deepseek_instance("Test prompt", only_completed=True)
        assert result == []

        result = deepseek_instance("Test prompt", only_completed=False)
        assert result == []
