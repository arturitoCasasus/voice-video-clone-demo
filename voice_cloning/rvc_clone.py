#!/usr/bin/env python3
"""
Example script for voice cloning using RVC (Retrieval-based Voice Conversion).
Note: This script assumes you have a pre-trained RVC model. You must first train a model
using your reference audio with the RVC training scripts (see https://github.com/RVC-Project/Retrieval-based-Voice-Conversion).
This script performs inference only: it converts a base audio (generated from text) to the target voice using RVC.
"""

import argparse
import os
import sys
import torch
import subprocess
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Clone voice using RVC (requires pre-trained model)")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    parser.add_argument("--reference_wav", type=str, required=True, help="Path to reference speaker audio (used for training, not inference)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained RVC model file (.pth)")
    parser.add_argument("--index_path", type=str, default="", help="Path to the index file (if using index-based search)")
    parser.add_argument("--f0_up_key", type=int, default=0, help="Pitch shift (integer, e.g., 0 for no change)")
    parser.add_argument("--index_rate", type=float, default=0.75, help="Influence of index (0 to 1)")
    parser.add_argument("--filter_radius", type=int, default=3, help="Filter radius for median filtering of f0")
    parser.add_argument("--resample_sr", type=int, default=0, help="Resample output audio to this sample rate (0 to keep original)")
    parser.add_argument("--rms_mix_rate", type=float, default=0.25, help="Mix ratio of RMS envelope")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect voiceless consonants and breath sounds (0 to 0.5)")
    args = parser.parse_args()

    # Check if rvc-python is available
    try:
        from rvc_python.infer import Infer
    except ImportError:
        print("Error: rvc-python is not installed. Please install it first.")
        print("You can install it with: pip install rvc-python")
        sys.exit(1)

    # We need to generate a base audio from text. We'll use Coqui TTS for this.
    # Check if TTS is available
    try:
        from TTS.api import TTS
    except ImportError:
        print("Error: TTS is not installed. Please install it first.")
        print("You can install it with: pip install TTS")
        sys.exit(1)

    # Create a temporary file for the base audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        base_audio_path = tmp_file.name

    try:
        # Step 1: Generate base audio from text using Coqui TTS (with a default voice)
        print("Generating base audio from text using Coqui TTS...")
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
        # Use a default speaker (we can use the first available speaker, but XTTS v2 is multi-speaker and needs a speaker_wav)
        # Since we don't have a reference for the base voice, we'll use a dummy speaker_wav (we can use the reference_wav? but that would bias)
        # Alternatively, we can use a model that doesn't require a speaker_wav for the base, but XTTS v2 does.
        # Let's use the reference_wav for the base audio generation? That would make the base audio already in the target voice.
        # But then we are not doing conversion. We want to show the conversion step.
        # We'll use a different approach: use a TTS model that doesn't require a speaker_wav for the base, like Tacotron2.
        # However, for simplicity and to avoid extra dependencies, we'll use the reference_wav for the base audio generation
        # and then convert it with RVC (which might be redundant but demonstrates the pipeline).
        # Actually, we can use the same reference_wav for the base audio generation (so the base audio is in the target voice)
        # and then convert it with RVC (which should ideally not change it much if the model is good).
        # Alternatively, we can use a neutral voice. Let's use the reference_wav for the base audio generation.
        tts.tts_to_file(text=args.text, speaker_wav=args.reference_wav, language="en", file_path=base_audio_path)
        print(f"Base audio saved to {base_audio_path}")

        # Step 2: Use RVC to convert the base audio to the target voice (using the trained model)
        print("Converting base audio to target voice using RVC...")
        infer = Infer(
            model_path=args.model_path,
            index_path=args.index_path if args.index_path else None,
            f0_up_key=args.f0_up_key,
            index_rate=args.index_rate,
            filter_radius=args.filter_radius,
            resample_sr=args.resample_sr,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        # Note: The rvc-python Infer class expects an input audio and returns the converted audio.
        # We'll call the infer method and save the output.
        audio_out = infer.infer(base_audio_path)
        # audio_out is a numpy array, we need to save it as wav
        import soundfile as sf
        sf.write(args.output, audio_out, infer.target_sr if hasattr(infer, 'target_sr') else 24000)
        print(f"Audio saved to {args.output}")

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary file
        if os.path.exists(base_audio_path):
            os.remove(base_audio_path)

if __name__ == "__main__":
    main()