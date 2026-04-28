# Demo assets

This directory intentionally contains **no real face photo or voice recording**.

To run the notebooks or Kaggle kernels, provide your own private assets outside git:

- `portrait.jpg`: a front-facing image of the person to animate.
- `reference_qwen_icl.wav`: clean reference voice audio.
- `reference_text.txt`: exact transcript of the reference audio.

Recommended local private path:

```text
assets/private/portrait.jpg
assets/private/reference_qwen_icl.wav
assets/private/reference_text.txt
```

Those paths are ignored by `.gitignore`.
