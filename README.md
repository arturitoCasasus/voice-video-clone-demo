# Voice and Video Cloning Demo

This repository contains examples and scripts for free, open-source voice cloning and talking head generation.

## Contents

- [Voice Cloning](#voice-cloning)
  - [Tortoise-TTS](#tortoise-tts)
  - [Coqui TTS](#coqui-tts)
- [Talking Head Generation](#talking-head-generation)
  - [Wav2Lip](#wav2lip)
  - [SadTalker](#sadtalker)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Video Duration Constraints](#video-duration-constraints)

## Voice Cloning

### Tortoise-TTS

Tortoise-TTS is a multi-voice TTS system trained with an emphasis on quality.

- Repository: https://github.com/neonbjb/tortoise-tts
- License: Apache License 2.0

### Coqui TTS

Coqui TTS is a deep learning toolkit for Text-to-Speech, battle-tested in research and production.

- Repository: https://github.com/coqui-ai/TTS
- License: MPL-2.0

## Talking Head Generation

### Wav2Lip

Wav2Lip is a lip sync expert that is remarkably robust to diverse voices, languages, and speaking styles.

- Repository: https://github.com/Rudrabha/Wav2Lip
- License: MIT

### SadTalker

SadTalker aims to generate talking faces from audio and a single face image.

- Repository: https://github.com/OpenTalker/SadTalker
- License: GPL-3.0

## Demo

We provide a demo that uses the provided avatar image and a test text to generate a cloned voice and a talking head video.

**Note**: The demo uses `demo/avatar.jpg` (provided by the user) as the avatar reference. Replace `demo/text.txt` with your desired text.

### Steps to run the demo:

1. Install the required dependencies (see [Installation](#installation)).
2. Run the voice cloning script to generate audio from text (ensure the audio duration matches the desired video length, see [Video Duration Constraints](#video-duration-constraints)).
3. Use the generated audio and the avatar image to create a talking head video.

## Installation

### Prerequisites

- Python 3.8+
- Git
- FFmpeg (for video processing)

### Cloning the repository

```bash
git clone https://github.com/arturitoCasasus/voice-video-clone-demo.git
cd voice-video-clone-demo
```

### Setting up the environment

We recommend using a virtual environment or conda.

#### Using venv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Installing dependencies

Each method has its own dependencies. We provide separate requirements files.

##### Tortoise-TTS

```bash
pip install tortoise-tts
```

##### Coqui TTS

```bash
pip install TTS
```

##### Wav2Lip

```bash
# First, clone Wav2Lip repo (if not already done)
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
# Download the pretrained model (check the repo for the latest link)
# Example: wget https://iiitaphyd-my.sharepoint.com/:u:/g/personal/rudrabha_iiit_ac_in/Ed8RfwqyBlVNmOubq6VukBsB6Xe7aKJxWgZQZ6y6vYgJcQ?download=1 -O wav2lip.pth
# Then install requirements
pip install -f https://download.pytorch.org/whl/torch_stable.html torch torchvision
pip install -r requirements.txt
cd ..
```

##### SadTalker

```bash
# Clone SadTalker repo
git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker
# Download pretrained models (check the repo for links)
# Example: wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapper.pth
#          wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/checkpoint.pth
pip install -r requirements.txt
cd ..
```

## Usage

### Voice Cloning with Tortoise-TTS

```bash
# From the root of this repository
python voice_cloning/tortoise_tts_clone.py --text "Your text here." --output audio/tortoise_output.wav
```

### Voice Cloning with Coqui TTS

```bash
python voice_cloning/coqui_tts_clone.py --text "Your text here." --output audio/coqui_output.wav --speaker_wav "path/to/reference/speaker.wav"
```

### Talking Head Generation with Wav2Lip

```bash
python video_generation/wav2lip_inference.py --face "demo/avatar.jpg" --audio "audio/tortoise_output.wav" --output "results/wav2lip_output.mp4"
```

### Talking Head Generation with SadTalker

```bash
python video_generation/sadtalker_inference.py --driven_audio "audio/tortoise_output.wav" --source_image "demo/avatar.jpg" --output_dir "results/sadtalker_output/"
```

## Demo Files

- `demo/avatar.jpg`: The provided avatar image to be used as the talking head.
- `demo/text.txt`: Sample text for voice cloning (edit to your desired content).

## Video Duration Constraints

The final video must be between **3 and 10 minutes** long.

To achieve this:

1. **Adjust the input text length**: The speaking rate varies, but a rough estimate is 130-150 words per minute for clear speech. For a 3‑minute video aim for ~400 words; for a 10‑minute video aim for ~1300 words.
2. **Control the TTS generation**:
   - With Tortoise-TTS, you can generate longer passages directly.
   - With Coqui TTS (XTTS v2), you can also pass long texts.
3. **If needed, concatenate audio clips**: Generate multiple clips and concatenate them with FFmpeg or Audacity to reach the exact duration.
4. **Trim or loop**: If the generated audio is too short, you can loop a segment (with slight variations to avoid monotony) or add pauses. If too long, trim to the desired length.
5. **Verify duration**: Use `ffprobe` to check audio length:
   ```bash
   ffprobe -i audio/output.wav -show_entries format=duration -v quiet -of csv="p=0"
   ```
   Then ensure the talking‑head video will have the same duration (the tools generally preserve audio length).

### Example: Generating a ~5‑minute narration (~650 words)

Edit `demo/text.txt` with your desired script (around 650 words). Then run the cloning script and proceed to video generation.

## Notes

- This repository is for educational purposes. Always respect privacy and copyright when using someone's voice or image.
- The quality of the output depends on the quality of the input data and the pretrained models used.
- For best results, use a clear front-facing photo (the provided avatar) and a clean audio recording.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Tortoise-TTS: https://github.com/neonbjb/tortoise-tts
- Coqui TTS: https://github.com/coqui-ai/TTS
- Wav2Lip: https://github.com/Rudrabha/Wav2Lip
- SadTalker: https://github.com/OpenTalker/SadTalker