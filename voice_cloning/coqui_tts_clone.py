#!/usr/bin/env python3
"""
Example script for voice cloning using Coqui TTS (XTTS v2).
"""

import argparse
import os
import sys
import torch

def main():
    parser = argparse.ArgumentParser(description="Clone voice using Coqui TTS")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    parser.add_argument("--speaker_wav", type=str, required=True, help="Path to the speaker reference audio file")
    parser.add_argument("--model_name", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2", help="Model name to use")
    parser.add_argument("--language", type=str, default="en", help="Language of the text")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Check if TTS is available
    try:
        from TTS.api import TTS
    except ImportError:
        print("Error: TTS is not installed. Please install it first.")
        print("You can install it with: pip install TTS")
        sys.exit(1)

    # Initialize the model
    print(f"Loading model {args.model_name}...")
    try:
        tts = TTS(model_name=args.model_name, progress_bar=False, gpu=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Generate audio
    print(f"Generating audio for text: {args.text}")
    try:
        tts.tts_to_file(text=args.text, speaker_wav=args.speaker_wav, language=args.language, file_path=args.output)
        print(f"Audio saved to {args.output}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()