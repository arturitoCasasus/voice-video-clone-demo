# Demo assets

This directory intentionally contains **no real face photo or voice recording**. `avatar_placeholder.svg` is a synthetic placeholder; do not replace it with a real person photo.

To run the notebooks or Kaggle kernels, provide your own private assets outside git:

- `portrait.jpg`: a front-facing image of the person to animate.
- `audio.wav`: generated or recorded speech for audio-driven talking-head models.
- `reference_voice.wav`: clean reference voice audio for Voicebox/Chatterbox candidate generation.
- `reference_qwen_icl.wav`: clean reference voice audio for Qwen ICL.
- `reference_text.txt`: exact transcript of the Qwen reference audio.

Recommended local private path:

```text
assets/private/portrait.jpg
assets/private/audio.wav
assets/private/reference_voice.wav
assets/private/reference_qwen_icl.wav
assets/private/reference_text.txt
```

Those paths are ignored by `.gitignore`.
