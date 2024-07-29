"""
Run video generation with command:
    poetry run python mind_renderer/core/video.py submit --prompt the prompt

Fetch generated video with command:
    poetry run python mind_renderer/core/video.py fetch --task_id runway_f2450ac24c1e
"""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional

import aiohttp
from dotenv import load_dotenv

from mind_renderer.utils.logger import Logger


class RunwayHTTPClient:
    def __init__(self, session: Optional[aiohttp.ClientSession] = None, api_key: str = "") -> None:
        """
        Initializes the HTTPClient with the URL and headers.

        Args:
            session (Optional[aiohttp.ClientSession]): An optional aiohttp ClientSession to use for requests.
        """
        self.base_url = "https://api.302.ai"
        self.logger = Logger(__name__)
        load_dotenv(override=True)
        self.api_key = api_key if api_key else os.getenv("302AI_API_KEY", "test_api_key")

        self.headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": "Apifox/1.0.0 (https://apifox.com)"}
        self.session = session

    async def send_post_request(self, prompt: str, seconds: int = 10, seed: str = "") -> Dict[str, Any]:
        """
        Sends an asynchronous HTTP POST request with the specified text prompt and other form data.

        Args:
            prompt (str): The text prompt to include in the request.
            seconds (int, optional): The duration in seconds. Defaults to 10.
            seed (str, optional): The seed value. Defaults to an empty string.

        Returns:
            Dict[str, Any]: The JSON response from the server.
        """
        url = f"{self.base_url}/runway/submit"
        data = {"text_prompt": prompt, "seconds": str(seconds), "seed": seed}

        if self.session:
            async with self.session.post(url, headers=self.headers, data=data) as response:
                return await response.json()
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, data=data) as response:
                    return await response.json()

    async def fetch_generated_video(self, task_id: str) -> str:
        """
        Fetches the generated video for a given task ID and saves it as an MP4 file.

        Args:
            task_id (str): The ID of the task to fetch the video for.
            session (Optional[aiohttp.ClientSession]): An optional aiohttp ClientSession to use for requests.

        Returns:
            str: The path to the saved MP4 file.
        """
        url = f"{self.base_url}/runway/task/{task_id}/fetch"

        generated_video_path = None
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                data = await response.read()
                decoded_data = data.decode("utf-8")
                response_json = json.loads(decoded_data)
                if "task" not in response_json:
                    self.logger.error(f"Error fetching generated video: {response_json}")
                    return
                task_blob = response_json["task"]
                if "status" not in task_blob:
                    self.logger.error(f"Error fetching generated video: {task_blob}")
                    return
                status = task_blob["status"]
                if status == "PENDING":
                    self.logger.info("Task is still pending...")
                    return
                elif status == "RUNNING":
                    self.logger.info("Task is still running...")
                    return
                elif status == "SUCCEEDED":
                    generated_video_path = task_blob["artifacts"][0]["url"]
                    self.logger.info(f"Generated video path: {generated_video_path}")
                else:
                    self.logger.error(f"Task failed with status: {status}, Response: {task_blob}")
                    return

        file_path = f"{task_id}.mp4"
        # Download mp4 from given url
        if generated_video_path:
            self.logger.info(f"Downloading generated video from: {generated_video_path}...")
            async with aiohttp.ClientSession() as session:
                async with session.get(generated_video_path) as response:
                    data = await response.read()
                    with open(file_path, "wb") as f:
                        f.write(data)
            self.logger.info(f"Generated video saved as: {file_path}")

        return file_path


def parse_arg() -> argparse.Namespace:
    """
    Parses the command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Runway API Client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit a new task")
    submit_parser.add_argument("--prompt", type=str, required=True, help="The text prompt to include in the request.")
    submit_parser.add_argument("--seconds", type=int, default=10, help="The duration in seconds.")
    submit_parser.add_argument("--seed", type=str, default="", help="The seed value.")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch a generated video")
    fetch_parser.add_argument("--task_id", type=str, required=True, help="The ID of the task to fetch the video for.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arg()
    client = RunwayHTTPClient()

    if args.command == "submit":
        response = asyncio.run(
            client.send_post_request(
                prompt=args.prompt,
                seconds=args.seconds,
                seed=args.seed,
            )
        )
        print(f"Response: {response}")
    elif args.command == "fetch":
        asyncio.run(client.fetch_generated_video(task_id=args.task_id))
