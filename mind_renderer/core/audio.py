import os
import time

import openai
from dotenv import load_dotenv
from pydub.utils import mediainfo
from tqdm import tqdm

load_dotenv(override=True)

# Replace with your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()


def transcribe_audio(file_path: str) -> str:
    # Open the audio file
    with open(file_path, "rb") as audio_file:
        # Use the OpenAI Whisper API to transcribe the audio
        response = openai.audio.transcriptions.create(
            file=audio_file, model="whisper-1", response_format="verbose_json", timestamp_granularities=["word"]
        )
        return response.text


if __name__ == "__main__":

    # Path to your audio file
    audio_file_path = "test.mp3"
    folder_path = os.path.expanduser("~/Downloads")
    audio_file_path = os.path.join(folder_path, audio_file_path)

    # Estimate the duration of the audio file
    audio_info = mediainfo(audio_file_path)
    duration = float(audio_info["duration"])

    # Simulate progress updates (since the actual API call is synchronous)
    print("Transcribing audio...")
    for i in tqdm(range(int(duration)), desc="Progress", unit="s"):
        time.sleep(1)  # Simulate time passing for each second of audio

    # Transcribe the audio file
    transcription = transcribe_audio(audio_file_path)

    # Save the transcription to a text file
    output_file_path = "output_transcription.txt"
    with open(folder_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    print("Transcription completed and saved to output_transcription.txt")
