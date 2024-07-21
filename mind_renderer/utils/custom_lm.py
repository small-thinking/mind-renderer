"""The wrapper to call Deepseek LM.
"""

import json
import logging
import os
from typing import Any, Literal, Optional

import backoff
import dspy
import requests

from mind_renderer.utils.config_loader import ConfigLoader
from mind_renderer.utils.logger import Logger


def init_lm(config_loader: ConfigLoader) -> dspy.LM:
    """Initialize the language model based on the configuration."""
    provider = config_loader.get_value("text_model.provider")
    lm_name = config_loader.get_value("text_model.lm_name")
    section_length = config_loader.get_value("text_model.section_length")
    if provider == "DeepSeek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        return DeepSeek(base_url="https://api.deepseek.com", model=lm_name, api_key=api_key, max_tokens=section_length)
    elif provider == "GROQ":
        api_key = os.getenv("GROQ_API_KEY")
    elif provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
    return dspy.__dict__[provider](model=lm_name, max_tokens=section_length, api_key=api_key)


class DeepSeek(dspy.LM):
    """Wrapper around DeepSeek's API.

    Args:
        model (str, optional): DeepSeek supported LLM model to use. Defaults to "deepseek-chat".
        api_key (Optional[str], optional): API provider Authentication token. Defaults to None.
        api_base (Optional[str], optional): The API base URL to use. Defaults to "https://api.deepseek.com".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Defaults to "chat".
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://api.deepseek.com",
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.logger = Logger(__name__)
        self.provider = "openai"
        self.api_base = api_base
        self.system_prompt = system_prompt
        self.model_type = model_type

        if api_key:
            self.api_key = api_key
        else:
            raise ValueError("API key must be provided")

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def _deepseek_client(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def log_usage(self, response):
        """Log the total tokens from the DeepSeek API response."""
        usage_data = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logging.debug(f"DeepSeek Response Token Usage: {total_tokens}")

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            payload = {"model": "deepseek-chat", "messages": messages, **kwargs}
            response = requests.post(
                f"{self.api_base}/chat/completions", headers=self._deepseek_client(), data=json.dumps(payload)
            )
        else:
            payload = {"model": "deepseek-chat", "prompt": prompt, **kwargs}
            response = requests.post(
                f"{self.api_base}/completions", headers=self._deepseek_client(), data=json.dumps(payload)
            )

        response_data = response.json()

        history = {
            "prompt": prompt,
            "response": response_data,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response_data

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException,),
        max_time=1000,
        on_backoff=lambda details: logging.warning(
            f"Backing off {details['wait']} seconds after {details['tries']} tries calling DeepSeek API"
        ),
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from DeepSeek API whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.model_type == "chat":
            return choice["message"]["content"]
        return choice["text"]

    def __call__(
        self,
        prompt: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from DeepSeek.

        Args:
            prompt (str): prompt to send to DeepSeek
            return_sorted (bool, optional): sort the completion choices using the returned probabilities.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """
        response = self.request(prompt, **kwargs)
        self.logger.error(f"DeepSeek Response: {response}")

        self.log_usage(response)
        choices = response["choices"]

        choices = [c for c in choices if c["finish_reason"] != "length"]

        if kwargs.get("logprobs", False):
            completions = [{"text": self._get_choice_text(c), "logprobs": c["logprobs"]} for c in choices]
        else:
            completions = [self._get_choice_text(c) for c in choices]

        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "" in tokens:
                    index = tokens.index("") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c), logprobs))
            scored_completions = sorted(scored_completions, reverse=True)
            if logprobs:
                completions = [{"text": c, "logprobs": lp} for _, c, lp in scored_completions]
            else:
                completions = [c for _, c in scored_completions]

        return completions
