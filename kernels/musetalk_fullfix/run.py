import json, os, pathlib, shutil, subprocess, sys, traceback, textwrap
WORK = pathlib.Path('/kaggle/working')
FINAL = WORK/'result_musetalk_fullfix.mp4'
MANIFEST = WORK/'manifest.json'
LOG_TXT = WORK/'musetalk_attempt_notes.txt'

def sh(cmd, **kw):
    printable = cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))
    print('\n$', printable, flush=True)
    return subprocess.check_call(cmd, **kw)

def out(cmd, **kw):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, **kw)

def find_asset(name):
    hits=[]
    for r in [pathlib.Path('/kaggle/input'), WORK, pathlib.Path('/kaggle/src')]:
        if r.exists(): hits.extend(r.rglob(name))
    if not hits: raise FileNotFoundError(name)
    print('asset', name, '->', hits[0], flush=True)
    return hits[0]

manifest={'engine':'MuseTalk v1.5 py3.10/MMLab full asset setup','final':str(FINAL),'errors':{}}
try:
    print('System Python:', sys.version)
    print('Inputs:', list(pathlib.Path('/kaggle/input').glob('*')) if pathlib.Path('/kaggle/input').exists() else None)
    sh(['apt-get','update','-qq'])
    sh(['apt-get','install','-y','-qq','ffmpeg','git','wget','libgl1','libglib2.0-0','python3.10','python3.10-dev','python3.10-venv'])

    face = find_asset('portrait.jpg')
    audio = find_asset('audio.wav')
    audio16 = WORK/'audio_16k.wav'
    sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(audio),'-ar','16000','-ac','1','-t','8',str(audio16)])
    manifest['audio_duration'] = out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(audio16)]).strip()

    os.chdir(WORK)
    sh(['git','clone','--depth','1','https://github.com/TMElyralab/MuseTalk.git'])
    repo = WORK/'MuseTalk'
    os.chdir(repo)

    venv = WORK/'py310'
    sh(['python3.10','-m','venv',str(venv)])
    py = str(venv/'bin/python')
    pip = [py,'-m','pip']
    sh(pip+['install','-U','pip','setuptools','wheel'])
    # Torch 2.0.1+cu118 is cp310-compatible and still supports Pascal/P100 (sm_60), unlike Kaggle's default torch 2.10+cu128.
    sh(pip+['install','--no-cache-dir','torch==2.0.1+cu118','torchvision==0.15.2+cu118','torchaudio==2.0.2+cu118','--index-url','https://download.pytorch.org/whl/cu118'])
    sh(pip+['install','--no-cache-dir','numpy==1.23.5','pillow==10.4.0','diffusers==0.30.2','accelerate==0.28.0','opencv-python==4.9.0.80','soundfile==0.12.1','transformers==4.39.2','huggingface_hub==0.30.2','librosa==0.11.0','einops==0.8.1','gdown','requests','imageio[ffmpeg]','omegaconf','ffmpeg-python','moviepy==1.0.3','scikit-image','tqdm'])
    sh(pip+['install','--no-cache-dir','openmim==0.3.9','mmengine==0.10.7'])
    # Official MuseTalk README pins these. Use mim under Python 3.10 after modern setuptools, avoiding Py3.12 ImpImporter failure.
    sh([py,'-m','mim','install','mmcv==2.0.1'])
    sh([py,'-m','mim','install','mmdet==3.1.0'])
    
    # mmpose==1.1.0 depends on chumpy, whose sdist build currently breaks under modern pip/build isolation.
    # MuseTalk inference does not need chumpy's SMPL path, so install mmpose without deps and add the runtime deps explicitly.
    sh(pip+['install','--no-cache-dir','json_tricks','munkres','scipy','matplotlib','xtcocotools'])
    sh(pip+['install','--no-cache-dir','--no-deps','mmpose==1.1.0'])
    # Provide a tiny chumpy stub so optional imports do not crash if mmpose imports SMPL utilities.
    site = out([py,'-c','import site; print(site.getsitepackages()[0])']).strip()
    chumpy_dir = pathlib.Path(site)/'chumpy'
    chumpy_dir.mkdir(exist_ok=True)
    (chumpy_dir/'__init__.py').write_text('''# lightweight Kaggle stub for optional mmpose SMPL/chumpy import\nimport numpy as _np\ndef array(*a, **k): return _np.array(*a, **k)\ndef zeros(*a, **k): return _np.zeros(*a, **k)\ndef ones(*a, **k): return _np.ones(*a, **k)\ndef concatenate(*a, **k): return _np.concatenate(*a, **k)\n''')
    (chumpy_dir.parent/'chumpy-0.70.dist-info').mkdir(exist_ok=True)
    (chumpy_dir.parent/'chumpy-0.70.dist-info'/'METADATA').write_text('Name: chumpy\nVersion: 0.70\n')
    (chumpy_dir.parent/'chumpy-0.70.dist-info'/'RECORD').write_text('')
    

    print(out([py,'-c','import torch, mmcv, mmpose, mmdet; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print(mmcv.__version__, mmpose.__version__, mmdet.__version__)']))
    try: print(out(['nvidia-smi'])[:2000])
    except Exception as e: print('nvidia-smi failed', e)

    sh(['bash','download_weights.sh'])
    # download_weights.sh is incomplete/flaky on Kaggle for this repo, so force all runtime assets directly
    # through huggingface_hub in the Python 3.10 venv and verify them before inference.
    download_py = WORK/'force_musetalk_assets.py'
    download_py.write_text("import pathlib, shutil\nfrom huggingface_hub import hf_hub_download, snapshot_download\nroot = pathlib.Path('models')\nroot.mkdir(exist_ok=True)\nfor filename in ['musetalkV15/musetalk.json', 'musetalkV15/unet.pth']:\n    hf_hub_download(repo_id='TMElyralab/MuseTalk', filename=filename, local_dir='models', local_dir_use_symlinks=False)\nfor filename in ['musetalk/musetalk.json', 'musetalk/pytorch_model.bin']:\n    hf_hub_download(repo_id='TMElyralab/MuseTalk', filename=filename, local_dir='models', local_dir_use_symlinks=False)\nhf_hub_download(repo_id='yzd-v/DWPose', filename='dw-ll_ucoco_384.pth', local_dir='models/dwpose', local_dir_use_symlinks=False)\nsdvae = root/'sd-vae'\nif sdvae.exists(): shutil.rmtree(sdvae)\nsnapshot_download(repo_id='stabilityai/sd-vae-ft-mse', local_dir=str(sdvae), local_dir_use_symlinks=False, allow_patterns=['config.json','diffusion_pytorch_model.bin','diffusion_pytorch_model.safetensors'])\nwhisper = root/'whisper'\nif whisper.exists(): shutil.rmtree(whisper)\nsnapshot_download(repo_id='openai/whisper-tiny', local_dir=str(whisper), local_dir_use_symlinks=False, allow_patterns=['config.json','preprocessor_config.json','pytorch_model.bin','model.safetensors'])\nrequired = {\n  'musetalkV15/musetalk.json': 100,\n  'musetalkV15/unet.pth': 100_000_000,\n  'musetalk/musetalk.json': 100,\n  'musetalk/pytorch_model.bin': 100_000_000,\n  'dwpose/dw-ll_ucoco_384.pth': 100_000_000,\n  'sd-vae/config.json': 100,\n  'whisper/config.json': 100,\n  'whisper/preprocessor_config.json': 100,\n}\nfor rel, min_size in required.items():\n    path = root/rel\n    if not path.exists() or path.stat().st_size < min_size:\n        size = path.stat().st_size if path.exists() else 'missing'\n        raise FileNotFoundError(f'{rel} missing or too small: {size}')\nif not ((whisper/'pytorch_model.bin').exists() or (whisper/'model.safetensors').exists()):\n    raise FileNotFoundError('whisper weights missing')\nif not ((sdvae/'diffusion_pytorch_model.bin').exists() or (sdvae/'diffusion_pytorch_model.safetensors').exists()):\n    raise FileNotFoundError('sd-vae weights missing')\nprint('VERIFIED_ASSETS')\nfor p in sorted(root.rglob('*')):\n    sp = str(p)\n    if p.is_file() and any(x in sp for x in ['musetalkV15','musetalk/','dwpose','sd-vae','whisper']):\n        print(p, p.stat().st_size)" + '\n')
    sh([py, str(download_py)])
    manifest['verified_assets'] = True
    custom = WORK/'musetalk_config.yaml'
    custom.write_text(f"""
task_0:
  video_path: {face}
  audio_path: {audio16}
  result_name: result_musetalk_raw.mp4
  bbox_shift: 0
""".strip()+"\n")
    result_dir = WORK/'musetalk_results'
    sh([py,'-m','scripts.inference','--inference_config',str(custom),'--result_dir',str(result_dir),'--unet_model_path','models/musetalkV15/unet.pth','--unet_config','models/musetalkV15/musetalk.json','--version','v15','--fps','25','--batch_size','4'])
    vids = list(result_dir.rglob('*.mp4'))
    print('videos:', vids)
    if not vids:
        raise RuntimeError('MuseTalk finished without creating mp4')
    raw = vids[0]
    # Re-encode for Telegram/browser compatibility.
    sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-c:a','aac','-shortest',str(FINAL)])
    manifest.update({'status':'ok','output':FINAL.name,'raw_output':str(raw),'duration_seconds':out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(FINAL)]).strip(),'bytes':FINAL.stat().st_size})
except Exception:
    manifest['status']='error'
    manifest['errors']['traceback']=traceback.format_exc()
    print(manifest['errors']['traceback'])
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    LOG_TXT.write_text('MuseTalk py3.10 attempt failed. See manifest/logs.\n')
    raise
finally:
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('MANIFEST:', MANIFEST.read_text())

# Clean heavy intermediates, keep result/manifest and diagnostic notes.
for p in [WORK/'MuseTalk', WORK/'py310', WORK/'musetalk_results', WORK/'audio_16k.wav']:
    if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
    elif p.exists(): p.unlink()
LOG_TXT.write_text('MuseTalk py3.10/MMLab full asset setup completed. Output: '+FINAL.name+'\n')
print('DONE files:', [p.name for p in WORK.iterdir() if p.is_file()])
