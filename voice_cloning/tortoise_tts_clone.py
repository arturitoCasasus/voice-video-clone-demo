#!/usr/bin/env python3
"""
Example script for voice cloning using Tortoise-TTS with reference audio.
"""

import argparse
import os
import sys
import torch

def main():
    parser = argparse.ArgumentParser(description="Clone voice using Tortoise-TTS with reference audio")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    parser.add_argument("--reference_wav", type=str, required=True, help="Path to reference speaker audio file")
    parser.add_argument("--preset", type=str, default="fast", choices=["ultra_fast", "fast", "standard", "high_quality"], help="Quality preset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Check if tortoise-tts is available
    try:
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_audio
    except ImportError:
        print("Error: tortoise-tts is not installed. Please install it first.")
        print("You can install it with: pip install tortoise-tts")
        sys.exit(1)

    # Initialize TTS
    print("Initializing Tortoise-TTS...")
    tts = TextToSpeech()

    # Load and process reference audio
    print(f"Loading reference audio from: {args.reference_wav}")
    try:
        # Load reference audio (22050 Hz is expected by Tortoise)
        reference_audio = load_audio(args.reference_wav, 22050)
        
        # Get conditioning latents from the reference audio
        print("Extracting conditioning latents from reference audio...")
        conditioning_latents = tts.get_conditioning_latents([reference_audio])
        
    except Exception as e:
        print(f"Error processing reference audio: {e}")
        sys.exit(1)

    # Generate audio using the conditioned latents
    print(f"Generating audio for text: {args.text}")
    try:
        # Use the conditioning latents for voice cloning
        audio = tts.tts_with_preset(
            args.text, 
            preset=args.preset,
            cond_latents=conditioning_latents
        )
        
        # Save the audio
        import torchaudio
        torchaudio.save(args.output, audio.squeeze(0).cpu(), 24000)
        print(f"Audio saved to {args.output}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()