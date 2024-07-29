"""Run the tests with command:
    poetry run pytest tests/core/test_video.py
"""

import os
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

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


@pytest.mark.asyncio
async def test_fetch_generated_image():
    api_key = "test_api_key"
    task_id = "runway_1234"
    expected_file_path = f"{task_id}.mp4"

    mock_data = b"fake mp4 data"

    with patch.dict(os.environ, {"302AI_API_KEY": api_key}):
        mock_response = AsyncMock()
        mock_response.read.return_value = mock_data

        mock_session = AsyncMock(spec=ClientSession)
        mock_session.get.return_value.__aenter__.return_value = mock_response

        client = RunwayHTTPClient(session=mock_session, api_key=api_key)

        with patch("builtins.open", mock_open()) as mock_file:
            file_path = await client.fetch_generated_image(task_id)

        expected_url = f"https://api.302.ai/runway/task/{task_id}/fetch"
        expected_headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "Apifox/1.0.0 (https://apifox.com)"}

        mock_session.get.assert_called_once_with(expected_url, headers=expected_headers)
        mock_file.assert_called_once_with(expected_file_path, "wb")
        mock_file().write.assert_called_once_with(mock_data)

        assert file_path == expected_file_path
