"""Kaggle script: Qwen3-TTS ICL voice clone + SadTalker talking-head video.

Expected private Kaggle dataset files:
- portrait.jpg
- reference_qwen_icl.wav
- reference_text.txt

This script intentionally contains no private assets. Keep the kernel private if using
real people, and only use voices/images you have permission to process.
"""
import json, os, pathlib, shutil, subprocess, sys, traceback

WORK = pathlib.Path('/kaggle/working')
FINAL = WORK / 'result_qwen_icl_sadtalker_demo.mp4'
VOICE = WORK / 'qwen_icl_voice.wav'
VOICE16 = WORK / 'qwen_icl_voice_16k.wav'
MANIFEST = WORK / 'manifest.json'

TARGET_TEXT = (
    "Hello, this is a longer English demo for an AI voice and video pipeline. "
    "The goal is to check whether the cloned voice stays consistent, whether the rhythm feels natural, "
    "and whether the generated face animation remains stable for a longer message. "
    "A practical assistant should preserve context, make useful decisions, and help move work forward without unnecessary noise. "
    "This sample is intentionally generic, so the repository can stay public without including private identity data."
)


def sh(cmd, **kw):
    print('\n$', cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)), flush=True)
    return subprocess.check_call(cmd, **kw)


def out(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def find_asset(name):
    hits = []
    for root in [pathlib.Path('/kaggle/input'), pathlib.Path('/kaggle/working'), pathlib.Path('/kaggle/src')]:
        if root.exists():
            hits.extend(root.rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    print('asset', name, '->', hits[0])
    return hits[0]


def patch_sadtalker_for_modern_kaggle():
    import site

    for base in site.getsitepackages():
        ft = pathlib.Path(base) / 'torchvision/transforms/functional_tensor.py'
        if ft.parent.exists() and not ft.exists():
            ft.write_text('from .functional import *\n')
            print('created functional_tensor shim', ft)

    patched = []
    for pyfile in pathlib.Path('.').rglob('*.py'):
        txt = pyfile.read_text(errors='ignore')
        ntxt = (txt
            .replace('np.VisibleDeprecationWarning', 'DeprecationWarning')
            .replace('np.float,', 'float,').replace('np.float)', 'float)').replace('np.float]', 'float]').replace('np.float ', 'float ')
            .replace('np.int,', 'int,').replace('np.int)', 'int)').replace('np.int]', 'int]').replace('np.int ', 'int ')
            .replace('np.bool,', 'bool,').replace('np.bool)', 'bool)').replace('np.bool]', 'bool]').replace('np.bool ', 'bool ')
        )
        if ntxt != txt:
            pyfile.write_text(ntxt)
            patched.append(str(pyfile))
    print('patched numpy legacy aliases in', len(patched), 'files')

    pre = pathlib.Path('src/face3d/util/preprocess.py')
    if pre.exists():
        txt = pre.read_text(errors='ignore')
        target = 'trans_params = np.array([w0, h0, s, t[0], t[1]])'
        repl = '''\n    try:\n        s = float(np.asarray(s).reshape(-1)[0])\n        t = np.asarray(t).reshape(-1)\n        tx = float(t[0]); ty = float(t[1])\n    except Exception:\n        tx, ty = t[0], t[1]\n    trans_params = np.array([float(w0), float(h0), s, tx, ty], dtype=np.float32)'''
        if target in txt:
            pre.write_text(txt.replace(target, repl))
            print('patched align_img trans_params scalar conversion')


def main():
    manifest = {
        'engine': 'Qwen3-TTS 0.6B Base ICL clone + SadTalker patched',
        'target_text': TARGET_TEXT,
        'target_text_chars': len(TARGET_TEXT),
        'errors': {},
    }
    try:
        print('Python', sys.version)
        print('input dirs', list(pathlib.Path('/kaggle/input').glob('*')) if pathlib.Path('/kaggle/input').exists() else None)
        sh(['apt-get', 'update', '-qq'])
        sh(['apt-get', 'install', '-y', '-qq', 'ffmpeg', 'git', 'wget', 'libgl1', 'libglib2.0-0', 'libsm6', 'libxext6', 'espeak-ng'])

        # P100-compatible setup used successfully on Kaggle. T4 also works.
        sh([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchaudio', 'torchvision', 'torchtext', 'torchdata'])
        sh([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--force-reinstall',
            'torch==2.5.1+cu121', 'torchvision==0.20.1+cu121', 'torchaudio==2.5.1+cu121',
            '--index-url', 'https://download.pytorch.org/whl/cu121'])
        sh([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--force-reinstall', 'numpy==1.26.4', 'pillow==11.3.0'])
        sh([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir',
            'qwen-tts>=0.0.5', 'soundfile', 'librosa>=0.10.0', 'scipy', 'huggingface_hub', 'safetensors',
            'transformers>=4.36.0,<=4.57.6', 'accelerate', 'tqdm', 'opencv-python', 'imageio',
            'imageio-ffmpeg', 'pydub', 'yacs', 'pyyaml', 'joblib', 'scikit-image', 'kornia',
            'face-alignment', 'gfpgan', 'facexlib', 'basicsr', 'realesrgan', 'ffmpeg-python'])

        import numpy as np
        import soundfile as sf
        import torch
        print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        manifest['cuda_available'] = bool(torch.cuda.is_available())
        manifest['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

        portrait = find_asset('portrait.jpg')
        ref = find_asset('reference_qwen_icl.wav')
        ref_text = find_asset('reference_text.txt').read_text(encoding='utf-8').strip()
        manifest['reference_text_chars'] = len(ref_text)
        manifest['reference_duration'] = out(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nk=1:nw=1', str(ref)]).strip()

        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
            device_map='cuda' if torch.cuda.is_available() else 'cpu',
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        wavs, sr = model.generate_voice_clone(
            text=TARGET_TEXT,
            language='English',
            ref_audio=str(ref),
            ref_text=ref_text,
            x_vector_only_mode=False,
            do_sample=True,
            top_k=20,
            top_p=0.8,
            temperature=0.7,
            repetition_penalty=1.1,
            max_new_tokens=4096,
        )
        arr = np.asarray(wavs[0], dtype='float32').squeeze()
        peak = max(1e-6, float(np.abs(arr).max()))
        if peak > 1:
            arr = arr / peak * 0.95
        sf.write(str(VOICE), arr, sr)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sh(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(VOICE), '-ar', '16000', '-ac', '1', '-af', 'loudnorm=I=-18:TP=-2:LRA=11', str(VOICE16)])
        manifest['voice_duration'] = out(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nk=1:nw=1', str(VOICE16)]).strip()

        os.chdir(WORK)
        sh(['git', 'clone', '--depth', '1', 'https://github.com/OpenTalker/SadTalker.git'])
        os.chdir(WORK / 'SadTalker')
        if pathlib.Path('scripts/download_models.sh').exists():
            sh(['bash', 'scripts/download_models.sh'])
        else:
            raise RuntimeError('SadTalker download script not found')
        patch_sadtalker_for_modern_kaggle()

        result_dir = WORK / 'sadtalker_results'
        # Do not use --enhancer gfpgan for long videos unless you have enough RAM.
        cmd = [sys.executable, 'inference.py', '--driven_audio', str(VOICE16), '--source_image', str(portrait),
               '--result_dir', str(result_dir), '--still', '--preprocess', 'full', '--size', '256']
        sh(cmd)
        vids = list(result_dir.rglob('*.mp4')) if result_dir.exists() else []
        if not vids:
            raise RuntimeError('SadTalker produced no mp4')
        best = max(vids, key=lambda p: p.stat().st_size)
        raw = WORK / 'result_raw.mp4'
        shutil.copy(best, raw)
        # Recode for Telegram/browser compatibility.
        sh(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(raw),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.1',
            '-crf', '23', '-preset', 'veryfast', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(FINAL)])
        manifest.update({
            'output': FINAL.name,
            'duration_seconds': out(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nk=1:nw=1', str(FINAL)]).strip(),
            'bytes': FINAL.stat().st_size,
        })
    except Exception:
        manifest['errors']['pipeline'] = traceback.format_exc()
        print(manifest['errors']['pipeline'])
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        raise
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(MANIFEST.read_text())

    for p in [WORK / 'SadTalker', WORK / 'sadtalker_results', VOICE16, WORK / 'result_raw.mp4']:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


if __name__ == '__main__':
    main()
