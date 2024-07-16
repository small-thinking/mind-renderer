"""The cli to generate the story.

Run the following command to generate the story:
    poetry run python mind_renderer/cli.py --config-path story_gen_config.yaml
"""

import argparse

from mind_renderer.core.story_generators import OneStepStoryGenerator
from mind_renderer.utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description="One-step story generator CLI")
    parser.add_argument("--config-path", type=str, default="config.yaml", help="Path to the config file")

    args = parser.parse_args()

    logger = Logger("StoryGeneratorCLI")

    story_generator = OneStepStoryGenerator(config_path=args.config_path)
    story = story_generator.generate()

    logger.info("Generated story:")
    print(story)


if __name__ == "__main__":
    main()
