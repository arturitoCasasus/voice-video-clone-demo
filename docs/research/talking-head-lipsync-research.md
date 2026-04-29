# Talking-head and lip-sync research notes

This document captures the working research state for single-image talking-head generation from a private portrait and private speech audio.

Private inputs and generated identity outputs are intentionally excluded from git.

## Scope

Input contract for the working Kaggle kernels:

```text
portrait.jpg   # private portrait, supplied via private Kaggle dataset
audio.wav      # private speech audio, supplied via private Kaggle dataset
```

The repository stores only reproducible code, templates and notes. It must not store real portraits, voice samples, generated videos or Kaggle output folders.

## Best strategies found

### 1. Ditto, expression scale 0.65, best visual baseline

Current best baseline: **Ditto with expression movement damped to 0.65**.

Why it worked best:

- Strong overall talking-head quality from a single portrait.
- Natural head movement and identity preservation.
- Reducing only expression intensity helped avoid exaggerated mouth deformation while keeping pose motion.
- H.264/yuv420p re-encoding made the result safe to send through Telegram/browser players.

Implementation:

- Script: [`../kernels/ditto_exp065/run.py`](../../kernels/ditto_exp065/run.py)
- Metadata template: [`../kernels/ditto_exp065/kernel-metadata.template.json`](../../kernels/ditto_exp065/kernel-metadata.template.json)
- Notebook wrapper: [`../../notebooks/ditto_exp065_kaggle.ipynb`](../../notebooks/ditto_exp065_kaggle.ipynb)

Key setting:

```python
'use_d_keys': {
    'exp': 0.65,
    'pitch': 1.0,
    'yaw': 1.0,
    'roll': 1.0,
    't': 1.0,
}
```

Practical recommendation: use this as the benchmark to beat when testing future models or parameter sweeps.

### 2. MuseTalk v1.5, full asset setup, working alternative

MuseTalk v1.5 also produced a clean final video when run with a controlled Python 3.10 environment and verified model assets.

Why it is useful:

- Modern lip-sync model family, worth keeping as a reproducible baseline.
- Works on Kaggle GPU once dependencies and weights are set up explicitly.
- Produces a clean single-output MP4, not a debug split-screen.

Implementation:

- Script: [`../kernels/musetalk_fullfix/run.py`](../../kernels/musetalk_fullfix/run.py)
- Metadata template: [`../kernels/musetalk_fullfix/kernel-metadata.template.json`](../../kernels/musetalk_fullfix/kernel-metadata.template.json)
- Notebook wrapper: [`../../notebooks/musetalk_fullfix_kaggle.ipynb`](../../notebooks/musetalk_fullfix_kaggle.ipynb)

Best setup strategy:

- Use a Python 3.10 venv inside Kaggle instead of relying on Kaggle Python 3.12 for MMLab dependencies.
- Use Torch `2.0.1+cu118` for Python 3.10 compatibility.
- Install MMLab stack with pinned versions:
  - `mmcv==2.0.1`
  - `mmdet==3.1.0`
  - `mmpose==1.1.0`
- Install `mmpose` without the fragile `chumpy` build dependency and provide a minimal optional-import stub.
- Force-download and verify all runtime assets before inference:
  - `musetalkV15/musetalk.json`
  - `musetalkV15/unet.pth`
  - `musetalk/musetalk.json`
  - `musetalk/pytorch_model.bin`
  - `dwpose/dw-ll_ucoco_384.pth`
  - `sd-vae` config and weights
  - `whisper` config, preprocessor and weights

Practical recommendation: keep MuseTalk as a second baseline and revisit when testing better portrait crops, audio padding, cheek width and blending parameters.

### 3. SadTalker, stable fallback

SadTalker remains a useful fallback because it is relatively simple once patched for current Kaggle environments.

Implementation already present:

- [`../../notebooks/qwen_icl_sadtalker_long_kaggle.ipynb`](../../notebooks/qwen_icl_sadtalker_long_kaggle.ipynb)
- [`../../notebooks/sadtalker_from_audio_kaggle.ipynb`](../../notebooks/sadtalker_from_audio_kaggle.ipynb)
- [`../../kernels/qwen_sadtalker_long/run.py`](../../kernels/qwen_sadtalker_long/run.py)

Practical recommendation: use it when robustness matters more than natural mouth quality.

## Reproducibility workflow

1. Create a private Kaggle dataset with only:

   ```text
   portrait.jpg
   audio.wav
   ```

2. Copy the relevant metadata template:

   ```bash
   cp kernels/ditto_exp065/kernel-metadata.template.json kernels/ditto_exp065/kernel-metadata.json
   # or
   cp kernels/musetalk_fullfix/kernel-metadata.template.json kernels/musetalk_fullfix/kernel-metadata.json
   ```

3. Edit:

   ```json
   "id": "<kaggle-username>/<kernel-slug>",
   "dataset_sources": ["<kaggle-username>/<private-assets-dataset>"]
   ```

4. Push from the kernel directory with the Kaggle CLI or local wrapper.

5. Download only the final MP4 if needed. Do not commit it.

## Output policy

Do not commit:

- `portrait.jpg`
- `audio.wav`
- reference voice files
- generated MP4s
- Kaggle `output-current/`, `error-current/` or downloaded run folders
- logs containing private dataset names or paths

Commit only:

- scripts
- notebook wrappers
- metadata templates
- general research notes
- safe placeholder assets
