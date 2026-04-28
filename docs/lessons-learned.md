# Lessons learned from the Kaggle voice/video pipeline

This repo was updated after several real Kaggle runs combining voice cloning and talking-head generation. Private user assets and generated identity outputs are intentionally excluded.

## What worked best

- **Voice cloning:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` via `qwen-tts`, using ICL mode.
- **Reference mode:** `x_vector_only_mode=False` with exact `ref_text` matching `ref_audio`.
- **Talking head:** SadTalker can work on Kaggle after compatibility patches.
- **Long video delivery:** recode the final MP4 to H.264/yuv420p for Telegram/browser compatibility.

## Qwen TTS notes

- `x_vector_only_mode=False` requires non-empty `ref_text`.
- `ref_text` must match the reference audio closely; otherwise quality drops.
- A clean ~30-45 second reference clip is a good practical baseline.
- For longer target text, raise `max_new_tokens`.

## SadTalker on modern Kaggle notes

Kaggle Python 3.12 and current package versions require patches for old SadTalker code:

- Install PyTorch explicitly, e.g. `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`.
- Pin `numpy==1.26.4` and `pillow==11.3.0`.
- Create `torchvision.transforms.functional_tensor` shim when needed for `basicsr`.
- Replace removed NumPy aliases:
  - `np.VisibleDeprecationWarning` → `DeprecationWarning`
  - `np.float` → `float`
  - `np.int` → `int`
  - `np.bool` → `bool`
- Patch SadTalker `align_img` so `s`, `t[0]`, `t[1]` are scalar floats before creating `trans_params`.

## GFPGAN enhancer caution

For short clips, `--enhancer gfpgan` can improve faces. For ~1 minute clips on Kaggle, it may run out of memory. If you see:

```text
Your notebook tried to allocate more memory than is available.
```

rerun without `--enhancer gfpgan`.

## Delivery compatibility

Some generated SadTalker MP4s use MPEG-4 video (`mpeg4`) and may appear blank in Telegram or some players. Recode before sharing:

```bash
ffmpeg -y -i input.mp4 \
  -c:v libx264 -pix_fmt yuv420p -profile:v high -level 4.1 \
  -crf 23 -preset veryfast \
  -c:a aac -b:a 128k -movflags +faststart \
  output_h264.mp4
```

## Privacy rule

Do not commit:

- real face photos
- real voice references
- generated videos with identifiable faces/voices
- Kaggle output folders
- logs containing private dataset paths or signed URLs
