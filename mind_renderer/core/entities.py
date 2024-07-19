"""Defines the entities of a story.
"""

import os
import time
from typing import List


class StoryPiece:
    """A piece is a part of a story.
    A story piece can optionally have an image, an audio, or a video.
    """

    def __init__(
        self,
        idx: int,
        text: str = None,
        thumbnail_gen_prompt: str = None,
        image_uri: str = None,
        audio_uri: str = None,
        video_uri: str = None,
    ):
        """Generate a story piece. The piece can have an image, an audio, or a video, associated with the text.

        Args:
            idx (int): The index of the piece in the story.
            text (str): The text of the piece. Defaults to None.
            thumbnail_gen_prompt (str, optional): The prompt for generating the thumbnail. Defaults to None.
            image_uri (str, optional): The uri of the image. Defaults to None.
            audio_uri (str, optional): The uri of the audio. Defaults to None.
            video_uri (str, optional): The uri of the video. Defaults to None.
        """
        self.idx = idx
        self.text = text
        self.thumbnail_gen_prompt = thumbnail_gen_prompt
        self.image_uri = image_uri
        self.audio_uri = audio_uri
        self.video_uri = video_uri

    def __str__(self):
        return self.text


class Story:
    """A story is a collection of story pieces."""

    def __init__(
        self,
        config: dict[str, any],
        title: str,
        genres: str,
        visual_style: str = None,
        audio_style: str = None,
        pieces: List[StoryPiece] = None,
    ):
        """

        Args:
            title (str): The title of the story.
            genres (str): The genres of the story.
            visual_style (str, optional): The visual style of the story, e.g. cartoon, simple color. Defaults to None.
            audio_style (str, optional): The audio style of the story, story telling. Defaults to None.
            pieces (List[StoryPiece], optional): The pieces of the story. Defaults to None.
        """
        # Created timestamp
        self.created_timestamp = str(int(time.time()))
        self.title = title
        self.genres = genres
        self.visual_style = visual_style
        self.audio_style = audio_style
        self.pieces = pieces or []
        story_folder_name = os.path.join(f"{self.created_timestamp}-{self.title}")
        self.story_folder_path = os.path.join(config.get("output_path", "outputs"), story_folder_name)
        # Create the story folder
        os.makedirs(self.story_folder_path, exist_ok=True)

    def add_piece(self, piece: StoryPiece):
        self.pieces.append(piece)

    def __str__(self):
        return "\n".join([f"Episode {idx}:\n{str(piece)}\n" for idx, piece in enumerate(self.pieces)])

    def to_markdown(self):
        file_path = os.path.join(self.story_folder_path, "story.md")
        with open(file_path, "w") as file:
            file.write(f"# {self.title}\n")
            file.write(f"### {self.genres}\n")
            for idx, piece in enumerate(self.pieces):
                file.write(f"## Episode {idx}\n")
                file.write(f"{piece.text}\n")
                if piece.image_uri:
                    file.write(f"![image](./{piece.image_uri})\n")
                if piece.audio_uri:
                    file.write(f"![audio](./{piece.audio_uri})\n")
                if piece.video_uri:
                    file.write(f"![video](./{piece.video_uri})\n")
                file.write("\n")
        return file_path
