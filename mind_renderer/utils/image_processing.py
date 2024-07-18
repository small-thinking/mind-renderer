import os

import requests


def download_and_save_image(url: str, save_path: str) -> str:
    """Download an image from a URL and save it to the specified folder path."""
    try:
        response = requests.get(url)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading image: {e}")
        return None

    return save_path
