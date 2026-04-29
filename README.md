# Voice and Video Cloning Demo

Safe, reproducible examples for voice cloning and talking-head generation with open-source tools.

This repository intentionally contains **no real personal face images, voice recordings, or generated identity outputs**. Bring your own private assets when running the notebooks.

## Current recommended pipelines

After testing several approaches, the best working talking-head strategies are:

1. **Ditto, expression scale 0.65**: current best visual/lip-sync baseline from `portrait.jpg` + `audio.wav`.
2. **MuseTalk v1.5 full asset setup**: working modern lip-sync alternative on Kaggle GPU.
3. **SadTalker**: stable fallback, especially useful when robustness matters more than mouth realism.
4. **FFmpeg H.264/yuv420p recoding** for Telegram/browser-compatible MP4 delivery.

Functional Kaggle notebooks are in [`notebooks/`](notebooks/):

- [`voicebox_candidates_kaggle.ipynb`](notebooks/voicebox_candidates_kaggle.ipynb): generate and compare voice candidates before video.
- [`voicebox_chatterbox_wav2lip_kaggle.ipynb`](notebooks/voicebox_chatterbox_wav2lip_kaggle.ipynb): animate the chosen Chatterbox/Voicebox audio with Wav2Lip.
- [`qwen_icl_wav2lip_kaggle.ipynb`](notebooks/qwen_icl_wav2lip_kaggle.ipynb): Qwen3-TTS ICL voice clone + Wav2Lip.
- [`qwen_icl_sadtalker_long_kaggle.ipynb`](notebooks/qwen_icl_sadtalker_long_kaggle.ipynb): Qwen3-TTS ICL voice clone + longer SadTalker video.
- [`sadtalker_from_audio_kaggle.ipynb`](notebooks/sadtalker_from_audio_kaggle.ipynb): animate a portrait from an existing audio file.
- [`ditto_exp065_kaggle.ipynb`](notebooks/ditto_exp065_kaggle.ipynb): best Ditto parameter strategy.
- [`musetalk_fullfix_kaggle.ipynb`](notebooks/musetalk_fullfix_kaggle.ipynb): MuseTalk v1.5 full dependency/model setup.

Script templates are in [`kernels/`](kernels/):

- [`kernels/voicebox_candidates/run.py`](kernels/voicebox_candidates/run.py)
- [`kernels/voicebox_chatterbox_wav2lip/run.py`](kernels/voicebox_chatterbox_wav2lip/run.py)
- [`kernels/qwen_icl_wav2lip/run.py`](kernels/qwen_icl_wav2lip/run.py)
- [`kernels/qwen_sadtalker_long/run.py`](kernels/qwen_sadtalker_long/run.py)
- [`kernels/ditto_exp065/run.py`](kernels/ditto_exp065/run.py)
- [`kernels/musetalk_fullfix/run.py`](kernels/musetalk_fullfix/run.py)

Run guide: [`docs/how-to-run-working-notebooks.md`](docs/how-to-run-working-notebooks.md).

Detailed research notes: [`docs/research/talking-head-lipsync-research.md`](docs/research/talking-head-lipsync-research.md).

## Private assets expected

Create a private Kaggle dataset, or keep files locally outside git, with:

```text
portrait.jpg
reference_qwen_icl.wav
reference_text.txt
```

For Ditto, MuseTalk, SadTalker-only and other audio-driven talking-head notebooks:

```text
portrait.jpg
audio.wav
```

For Voicebox candidate generation:

```text
reference_voice.wav
```

For Voicebox/Chatterbox + Wav2Lip:

```text
portrait.jpg
voicebox_chatterbox_turbo_en.wav
```

Do **not** commit these assets. `.gitignore` excludes common audio/video outputs and private asset folders.

## Kaggle setup

Copy the template metadata:

```bash
cp kernels/qwen_sadtalker_long/kernel-metadata.template.json kernel-metadata.json
```

Then edit:

```json
"id": "<kaggle-username>/qwen-icl-sadtalker-long-demo",
"dataset_sources": ["<kaggle-username>/<private-assets-dataset>"]
```

Push with the Kaggle CLI or your wrapper:

```bash
kaggle kernels push
```

Recommended accelerator: GPU. The pipeline was tested with a Kaggle **Tesla P100** using explicit PyTorch `cu121` wheels.

## Important implementation notes

### Qwen ICL voice cloning

Use exact reference text:

```python
model.generate_voice_clone(
    text=TARGET_TEXT,
    language="English",
    ref_audio="reference_qwen_icl.wav",
    ref_text=reference_text,
    x_vector_only_mode=False,
)
```

`x_vector_only_mode=False` requires `ref_text`. The transcript should match the reference audio closely.

### SadTalker on modern Kaggle

SadTalker is old enough that modern Kaggle needs compatibility patches. The notebooks/scripts include fixes for:

- PyTorch/P100 compatibility.
- `torchvision.transforms.functional_tensor` removal.
- removed NumPy aliases such as `np.float`, `np.int`, `np.bool`.
- stricter NumPy scalar handling in SadTalker face alignment.

### Avoid GFPGAN for long clips

`--enhancer gfpgan` may work for short clips, but can run out of memory for ~1 minute videos. The long notebook disables it by default.

### Recode output video

Some generated MP4s may use `mpeg4` and display incorrectly in chat apps. Recode to H.264/yuv420p:

```bash
ffmpeg -y -i input.mp4 \
  -c:v libx264 -pix_fmt yuv420p -profile:v high -level 4.1 \
  -crf 23 -preset veryfast \
  -c:a aac -b:a 128k -movflags +faststart \
  output_h264.mp4
```

## Repository layout

```text
demo/                         Safe placeholder assets/text only
notebooks/                    Functional Kaggle notebooks
kernels/voicebox_candidates/  Voice candidate generation script + metadata template
kernels/voicebox_chatterbox_wav2lip/  Voicebox audio + Wav2Lip script + metadata template
kernels/qwen_icl_wav2lip/     Qwen ICL + Wav2Lip script + metadata template
kernels/ditto_exp065/         Best Ditto talking-head script + metadata template
kernels/musetalk_fullfix/     Working MuseTalk v1.5 script + metadata template
kernels/qwen_sadtalker_long/  Kaggle script + metadata template
docs/how-to-run-working-notebooks.md  Run guide for all working notebooks
docs/research/                Research notes from working runs
docs/lessons-learned.md       Practical notes from real runs
voice_cloning/                Older local examples
video_generation/             Older local examples
```

## Safety and consent

Only clone or animate voices/faces you own or have explicit permission to use. Keep private datasets and kernels private when processing identifiable people.

## License

MIT. See [`LICENSE`](LICENSE).
