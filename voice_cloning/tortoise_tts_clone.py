#!/usr/bin/env python3
"""
Example script for voice cloning using Tortoise-TTS.
"""

import argparse
import os
import sys

# Note: Tortoise-TTS requires the repository to be cloned and set up.
# For simplicity, we assume the user has installed tortoise-tts via pip or has set up the environment.

def main():
    parser = argparse.ArgumentParser(description="Clone voice using Tortoise-TTS")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    parser.add_argument("--preset", type=str, default="fast", choices=["ultra_fast", "fast", "standard", "high_quality"], help="Quality preset")
    args = parser.parse_args()

    # Check if tortoise-tts is available
    try:
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_audio, load_voice
        import torch
    except ImportError:
        print("Error: tortoise-tts is not installed. Please install it first.")
        print("You can install it with: pip install tortoise-tts")
        sys.exit(1)

    # Initialize TTS
    tts = TextToSpeech()

    # We don't have a specific voice to clone, so we use a random voice or a preset voice.
    # For voice cloning, you would need to provide a voice sample and use the `load_voice` function or fine-tune.
    # This example uses a preset voice for demonstration.
    # For actual voice cloning, refer to the Tortoise-TTS documentation.

    # Generate audio
    print(f"Generating audio for text: {args.text}")
    try:
        # Using a preset voice (e.g., 'rand' for random, or a specific voice like 'train')
        # For cloning, you would use a custom voice directory.
        # Here we use the default voice for simplicity.
        audio = tts.tts_with_preset(args.text, preset=args.preset)
        
        # Save the audio
        import torchaudio
        torchaudio.save(args.output, audio.squeeze(0).cpu(), 24000)
        print(f"Audio saved to {args.output}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()