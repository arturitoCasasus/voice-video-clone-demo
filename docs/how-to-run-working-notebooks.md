# How to run the working Kaggle notebooks

All notebooks expect private Kaggle datasets. Do not commit private photos, voice references or generated videos.

## General workflow

1. Pick the notebook/kernel you want to run.
2. Create a private Kaggle dataset with the required files.
3. Copy the metadata template:

   ```bash
   cp kernels/<kernel-name>/kernel-metadata.template.json kernels/<kernel-name>/kernel-metadata.json
   ```

4. Edit `id` and `dataset_sources` in `kernel-metadata.json`.
5. Push from that kernel folder:

   ```bash
   kaggle kernels push
   # or in this local setup:
   kaggle-oc kernels push
   ```

6. Download outputs only when needed. Keep them outside git.

## 1. Voicebox candidates

Notebook:

```text
notebooks/voicebox_candidates_kaggle.ipynb
```

Script:

```text
kernels/voicebox_candidates/run.py
```

Private dataset files:

```text
reference_voice.wav
```

Purpose:

Generate several candidate cloned voices before spending GPU time on video.

Expected useful output:

```text
candidate_chatterbox_turbo_en.wav
candidate_chatterbox_turbo_en.mp3
candidate_chatterbox_multilingual_es.wav
candidate_chatterbox_multilingual_es.mp3
manifest.json
```

Reflection:

Use this as the first step when voice quality matters. Listen to candidates manually, pick the best, then feed it into a video model.

## 2. Voicebox/Chatterbox + Wav2Lip

Notebook:

```text
notebooks/voicebox_chatterbox_wav2lip_kaggle.ipynb
```

Script:

```text
kernels/voicebox_chatterbox_wav2lip/run.py
```

Private dataset files:

```text
portrait.jpg
voicebox_chatterbox_turbo_en.wav
```

Purpose:

Animate the portrait using the selected Voicebox/Chatterbox candidate audio.

Reflection:

Good for testing the generated voice in a talking-head video, but Wav2Lip is not the strongest face animation model.

## 3. Qwen ICL + Wav2Lip

Notebook:

```text
notebooks/qwen_icl_wav2lip_kaggle.ipynb
```

Script:

```text
kernels/qwen_icl_wav2lip/run.py
```

Private dataset files:

```text
portrait.jpg
reference_qwen_icl.wav
reference_text.txt
```

Purpose:

Generate Qwen ICL cloned audio and immediately create a Wav2Lip video.

Reflection:

Better voice similarity than x-vector mode when the reference transcript is accurate. Video quality remains Wav2Lip-limited.

## 4. Qwen ICL + SadTalker long video

Notebook:

```text
notebooks/qwen_icl_sadtalker_long_kaggle.ipynb
```

Script:

```text
kernels/qwen_sadtalker_long/run.py
```

Private dataset files:

```text
portrait.jpg
reference_qwen_icl.wav
reference_text.txt
```

Purpose:

Generate a longer voice clone with Qwen ICL and animate it with SadTalker.

Reflection:

Stable for longer clips. Use when robustness matters more than top visual quality.

## 5. SadTalker from existing audio

Notebook:

```text
notebooks/sadtalker_from_audio_kaggle.ipynb
```

Private dataset files:

```text
portrait.jpg
audio.wav
```

Purpose:

Animate a portrait from already-generated or recorded audio.

Reflection:

Simple and useful as a fallback baseline.

## 6. Ditto exp065

Notebook:

```text
notebooks/ditto_exp065_kaggle.ipynb
```

Script:

```text
kernels/ditto_exp065/run.py
```

Private dataset files:

```text
portrait.jpg
audio.wav
```

Purpose:

Run the best observed talking-head baseline.

Key setting:

```python
'exp': 0.65
```

Reflection:

Best visual result so far. Use as the benchmark for future work.

## 7. MuseTalk fullfix

Notebook:

```text
notebooks/musetalk_fullfix_kaggle.ipynb
```

Script:

```text
kernels/musetalk_fullfix/run.py
```

Private dataset files:

```text
portrait.jpg
audio.wav
```

Purpose:

Run MuseTalk v1.5 with a deterministic Kaggle environment and verified model assets.

Reflection:

Promising modern baseline. More fragile setup than Ditto, but now reproducible.
