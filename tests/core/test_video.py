"""Run the tests with command:
    poetry run pytest tests/core/test_video.py
"""

import os
from unittest.mock import AsyncMock, mock_open, patch

import pytest
from aiohttp import ClientSession

from mind_renderer.core.video import RunwayHTTPClient


@pytest.mark.asyncio
async def test_send_post_request_success():
    url = "https://api.302.ai/runway/submit"
    api_key = "test_api_key"

    prompt = (
        "Low-angle shot of a majestic tiger prowling through a snowy landscape, leaving paw prints on the white blanket"
    )
    seconds = 10
    seed = ""

    expected_response = {"status": "success", "message": "Request submitted successfully"}

    mock_response = AsyncMock()
    mock_response.json.return_value = expected_response

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.post.return_value.__aenter__.return_value = mock_response

    client = RunwayHTTPClient(session=mock_session, api_key=api_key)
    response = await client.send_post_request(prompt, seconds, seed)

    expected_headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "Apifox/1.0.0 (https://apifox.com)"}
    mock_session.post.assert_called_once_with(
        url, headers=expected_headers, data={"text_prompt": prompt, "seconds": str(seconds), "seed": seed}
    )
    assert response == expected_response
