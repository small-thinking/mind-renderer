"""The cli to generate the story.

Run the following command to generate the story:
    poetry run python mind_renderer/cli.py
"""

import argparse

from mind_renderer.core.story_generators import OneStepStoryGenerator
from mind_renderer.utils.logger import Logger


def main():
    story_generator = OneStepStoryGenerator()
    story = story_generator.generate()

    print("Generated story:")
    story.to_markdown()


if __name__ == "__main__":
    main()
