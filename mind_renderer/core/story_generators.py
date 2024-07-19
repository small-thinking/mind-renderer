"""Story generator is in charge of generating the story, the narrative, and the visual/audio generation guidelines
according to the requirement of the user. It will be the main interface for the user to interact with the system.
"""

import json
import os
from abc import abstractmethod

import dspy
import retrying
from dotenv import load_dotenv
from openai import OpenAI

from mind_renderer.core.entities import Story, StoryPiece
from mind_renderer.utils.config_loader import ConfigLoader
from mind_renderer.utils.custom_lm import init_lm
from mind_renderer.utils.image_processing import download_and_save_image
from mind_renderer.utils.logger import Logger


class StorySketchGenerator(dspy.Module):
    """StorySketchGenerator is used to expand a simple prompt into detailed piece-level prompts."""

    instruction: str = """
        You are the story sketching assistant. Please follow the below instruction:

        0. Generate a sketch for a multi-section story according to the specified number of sections.
        1. Generate the sketch in the SAME language as the user input,
            e.g. generate Chinese sketch if user input is Chinese.
        2. Generate the story_worldview and the detailed piece-level prompts based on the simple prompt.
            The story_worldview is the high-level description of the story, used to make each piece consistent.
            Each of the detailed piece-level prompts will be used to generate a piece of the story.
        3. Each section should be coherent and complementary of the others.
        4. The story in combination should be complete. So the story pieces can be logically connected.
        5. Generate the prompt per section in the same language as the input prompt.
    """

    class SketchGenerateSignature(dspy.Signature):
        """Signature for the generate method."""

        instruction: str = dspy.InputField(desc="The instruction for the generation.")
        simple_prompt: str = dspy.InputField(desc="The user instruction on how the story would look like.")
        num_sections: str = dspy.InputField(
            desc="""
                The number of sections the story should have. Each section describe one scene.
                Consequtive sections should be coherent and complementary.
            """
        )
        story_title: str = dspy.OutputField(
            desc="The title of the story. Should be short and catchy. In the same language as the input prompt."
        )
        story_worldview: str = dspy.OutputField(desc="The world view of the story. Used to ensure consistency.")
        prompts: str = dspy.OutputField(
            desc="""
            The blob that contains a list of prompts for each piece of the story, they should be coherent, e.g.
            {
                "prompts": ["prompt1", "prompt2", ...]
            }
            """
        )

    def __init__(self, config: dict[str, any], text_model_config: dict[str, str]):
        self.logger = Logger(__name__)
        self.text_model_config = text_model_config
        self.lm = init_lm(self.text_model_config)
        self.sketch_inspect_length = int(os.getenv("SKETCH_INSPECT_LENGTH", 0))

    def forward(self, simple_prompt: str, num_sections: int, **kwargs):
        """Generate the detailed piece-level prompts based on the simple prompt."""
        return self.generate(simple_prompt=simple_prompt, num_sections=str(num_sections), **kwargs)

    def generate(self, simple_prompt: str, num_sections: int, **kwargs):
        """Generate the detailed piece-level prompts based on the simple prompt."""
        sketch_gen = dspy.Predict(self.SketchGenerateSignature)
        with dspy.context(lm=self.lm):
            response = sketch_gen(instruction=self.instruction, simple_prompt=simple_prompt, num_sections=num_sections)
            if self.sketch_inspect_length:
                self.lm.inspect_history(n=self.sketch_inspect_length)
            return response


class TextGenerator(dspy.Module):
    """TextGenerator is used to generate the text element of the story piece based on the prompt."""

    instructions = """
    You are the story writing assistant. Your goal is to generate a story piece a time based on the prompt.
    Please generate the story piece in the SAME language as the input prompt.
    """

    class TextGenerateSignature(dspy.Signature):
        """Signature for the generate method."""

        instruction: str = dspy.InputField(desc="The instruction for the generation.")
        story_description: str = dspy.InputField(desc="The description of the story at the high level.")
        story_worldview: str = dspy.InputField(desc="The story_worldview of the story.")
        story_genres: str = dspy.InputField(desc="The genres of the story, e.g. horror, romance, etc.")
        story_writing_style: str = dspy.InputField(desc="The writing style of the story, e.g. formal, informal, etc.")
        should_gen_thumbnail_prompt: bool = dspy.InputField(
            desc="Whether to generate the thumbnail image gneeration prompt."
        )
        prev_piece: str = dspy.InputField(desc="The previous piece of the story. For continuity.")

        story_text: str = dspy.OutputField(
            desc="The actual story content. Use the same language as the story description."
        )  # The output text of the story.
        thumbnail_generation_prompt: str = dspy.OutputField(
            desc="The cohensive prompt for generating the thumbnail image in English only. Use descriptive keywords."
        )

    def __init__(self, config: dict[str, any], text_model_config: dict[str, str]):
        self.logger = Logger(__name__)
        self.config = config
        self.text_model_config = text_model_config
        self.lm = init_lm(self.text_model_config)
        self.genres = config["genres"]
        self.writing_style = config["writing_style"]
        self.gen_thumbnail_prompt = config["gen_thumbnail"]

    def forward(
        self, prompt: str, story_worldview: str, story_piece: StoryPiece, prev_piece: StoryPiece = None, **kwargs
    ) -> None:
        """Generate the element based on the prompt and populate the story piece with the generated element."""
        self.generate(
            story_description=prompt,
            story_worldview=story_worldview,
            story_piece=story_piece,
            prev_piece=prev_piece,
            **kwargs,
        )

    @retrying.retry(
        wait_fixed=5000,
        stop_max_attempt_number=3,
    )
    def generate(
        self,
        story_description: str,
        story_worldview: str,
        story_piece: StoryPiece,
        prev_piece: StoryPiece = None,
        **kwargs,
    ) -> None:
        try:
            self.logger.info("Attempting to generate story piece...")
            story_gen = dspy.ChainOfThoughtWithHint(self.TextGenerateSignature)

            with dspy.context(lm=self.lm):
                response = story_gen(
                    instruction=self.instructions,
                    story_description=story_description,
                    story_worldview=story_worldview,
                    story_genres=self.genres,
                    story_writing_style=self.writing_style,
                    should_gen_thumbnail_prompt=str(self.gen_thumbnail_prompt),
                    prev_piece=prev_piece.text if prev_piece else "",
                )
                story_piece.text = response.story_text
                if not story_piece.text:
                    self.logger.error("Failed to generate the text for the story piece.")
                    raise ValueError(f"Failed to generate the text for the story piece, {response}")
                else:
                    story_piece.thumbnail_gen_prompt = response.thumbnail_generation_prompt
                    self.logger.info("Successfully generated story piece.")
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            raise


class ThumbnailGenerator(dspy.Module):
    """ThumbnailGenerator is used to generate thumbnail images for story pieces."""

    def __init__(self, config: dict[str, any]):
        self.logger = Logger(__name__)
        self.config = config
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.root_save_folder = config.get("root_save_folder", "outputs")
        self.save_folder = config.get("thumbnail_save_folder", "thumbnails")

    def forward(self, story_piece: StoryPiece, story: Story) -> None:
        """Generate the thumbnail for the given story piece."""
        self.generate(story_piece=story_piece, story=story)

    def generate(self, story_piece: StoryPiece, story: Story) -> None:
        """Generate the thumbnail based on the prompt in the story piece."""

        genres = self.config.get("genres", "")
        enhanced_prompt = f"""
        Generate a thumbnail image for the story piece. The story piece is part of a whole story.
        The genres of the story are: {genres}.
        The description of the thumbnail: {story_piece.thumbnail_gen_prompt}.
        """

        response = self.client.images.generate(
            model="dall-e-2",
            prompt=enhanced_prompt,
            size="512x512",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        thumbnail_name = f"thumbnail_{story_piece.idx}.png"
        save_path = os.path.join(story.story_folder_path, thumbnail_name)

        downloaded_path = download_and_save_image(image_url, save_path)
        story_piece.image_uri = thumbnail_name

        self.logger.info(f"Generated thumbnail for story piece {story_piece.idx}, saved at: {downloaded_path}")


class StoryGenerator(dspy.Module):
    """StoryGenerator is the main interface for the user to interact with the system.
    It generates the entire story based on the prompt and the previous version of the story.
    """

    def __init__(
        self,
        genres: str,
        writing_style: str,
        gen_thumbnail_prompt: bool,
        provider: str = None,
        lm_name: str = None,
        **kwargs,
    ):
        load_dotenv(override=True)
        self.state: dict[str, any]
        self.logger = Logger(__name__)
        self.genres = genres
        self.writing_style = writing_style
        self.gen_thumbnail_prompt = gen_thumbnail_prompt
        self.provider = provider
        self.lm_name = lm_name
        self.temperature = kwargs.get("temperature", 0.7)

    @abstractmethod
    def generate(self, prev_version: Story, **kwargs) -> Story:
        """Generate the story based on the prompt and the previous version of the story.

        Args:
            prev_version (Story): The previous version of the story.
        """
        pass

    @abstractmethod
    def _stop_condition_met(self) -> bool:
        """Define the stop condition of the story generation."""
        pass

    def forward(self, **kwargs):
        """The main logic of the story generation. It can be one-shot or interactive."""
        story = None
        while True:
            story = self.generate(prev_version=None, **kwargs)
            if self._stop_condition_met():
                break
        return story


class OneStepStoryGenerator(StoryGenerator):
    def __init__(self, config_path: str):
        self.logger = Logger(__name__)
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        text_model_config = self.config_loader.get_text_model_config()

        super().__init__(
            genres=self.config.get("genres", ""),
            writing_style=self.config.get("writing_style", ""),
            gen_thumbnail_prompt=self.config.get("gen_thumbnail", False),
            provider=text_model_config.get("provider"),
            lm_name=text_model_config.get("lm_name"),
        )

        self.story_sketch_generator = StorySketchGenerator(config=self.config, text_model_config=text_model_config)
        self.text_generator = TextGenerator(config=self.config, text_model_config=text_model_config)
        self.thumbnail_generator = ThumbnailGenerator(config=self.config)

    def generate(self, prev_version: Story = None, **kwargs) -> Story:
        config = self.config_loader.get_config()
        prompt = config.get("story", "")
        num_sections = config.get("num_sections", 4)

        self.logger.tool_log(f"Generating a {num_sections} pieces story with prompt: {prompt}...")

        # Generate story sketch
        self.logger.tool_log("Generating story sketch...")
        sketch_response = self.story_sketch_generator(simple_prompt=prompt, num_sections=num_sections)

        story_title = sketch_response.story_title.replace("Story Title:", "").strip()
        story_worldview = sketch_response.story_worldview
        try:
            prompts = json.loads(sketch_response.prompts)["prompts"]
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode the prompts: {sketch_response.prompts}")

        # Generate the story pieces
        story = Story(config=self.config, title=story_title, genres=self.genres)
        prev_story_piece = None
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Generating story piece {i+1} with prompt: {prompt}...")
            story_piece = StoryPiece(idx=i)
            self.text_generator(
                prompt=prompt,
                story_worldview=story_worldview,
                story_piece=story_piece,
                prev_piece=prev_story_piece,
            )
            # Generate thumbnail if enabled
            if self.gen_thumbnail_prompt:
                self.thumbnail_generator(story_piece=story_piece, story=story)
            story.add_piece(story_piece)

        return story

    def _stop_condition_met(self) -> bool:
        return True
