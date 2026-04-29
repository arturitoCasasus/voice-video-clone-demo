# Lessons learned from the Kaggle voice/video pipeline

This repo was updated after several real Kaggle runs combining voice cloning and talking-head generation. Private user assets and generated identity outputs are intentionally excluded.

## What worked best

- **Best talking-head baseline:** Ditto with expression movement scaled to `0.65`, while keeping pose/head movement at `1.0`.
- **Best modern lip-sync alternative:** MuseTalk v1.5 with Python 3.10, pinned MMLab stack and verified model assets.
- **Voice cloning:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` via `qwen-tts`, using ICL mode.
- **Reference mode:** `x_vector_only_mode=False` with exact `ref_text` matching `ref_audio`.
- **Stable fallback talking head:** SadTalker can work on Kaggle after compatibility patches.
- **Long video delivery:** recode the final MP4 to H.264/yuv420p for Telegram/browser compatibility.

## Qwen TTS notes

- `x_vector_only_mode=False` requires non-empty `ref_text`.
- `ref_text` must match the reference audio closely; otherwise quality drops.
- A clean ~30-45 second reference clip is a good practical baseline.
- For longer target text, raise `max_new_tokens`.

## Ditto exp065 notes

Ditto produced the strongest talking-head baseline in the current research pass.

Best parameter strategy:

```python
'use_d_keys': {
    'exp': 0.65,
    'pitch': 1.0,
    'yaw': 1.0,
    'roll': 1.0,
    't': 1.0,
}
```

Rationale:

- `exp=0.65` dampens mouth/expression artifacts.
- Pose keys at `1.0` preserve natural movement.
- A 20-second clipped test is enough for quick model comparison.
- Recode the raw output to H.264/yuv420p before sending or archiving.

## MuseTalk v1.5 notes

MuseTalk v1.5 works on Kaggle GPU when the environment and assets are controlled explicitly.

Best setup strategy:

- Create a Python 3.10 venv inside Kaggle.
- Use `torch==2.0.1+cu118`, `torchvision==0.15.2+cu118`, `torchaudio==2.0.2+cu118`.
- Pin MMLab stack:
  - `mmcv==2.0.1`
  - `mmdet==3.1.0`
  - `mmpose==1.1.0`
- Install `mmpose` without pulling the fragile `chumpy` build dependency; provide a small optional-import stub.
- Force-download and verify all runtime assets before inference, including `musetalkV15`, DWPose, SD VAE and Whisper.

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
