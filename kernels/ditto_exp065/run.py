import json, os, pathlib, shutil, subprocess, sys, traceback
WORK=pathlib.Path('/kaggle/working')
MANIFEST=WORK/'manifest.json'
FINAL=WORK/'result_ditto_exp065_h264.mp4'

def sh(cmd, **kw):
    print('\n$', cmd if isinstance(cmd,str) else ' '.join(map(str,cmd)), flush=True)
    return subprocess.check_call(cmd, **kw)

def out(cmd): return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def find_asset(name):
    hits=[]
    for r in [pathlib.Path('/kaggle/input'), pathlib.Path('/kaggle/working'), pathlib.Path('/kaggle/src')]:
        if r.exists(): hits.extend(r.rglob(name))
    if not hits: raise FileNotFoundError(name)
    print('asset', name, '->', hits[0])
    return hits[0]

manifest={'engine':'Ditto PyTorch lip-sync, expression scale 0.65', 'errors':{}, 'test_seconds':20, 'variant':'use_d_keys exp 0.65 / pose 1.0'}
try:
    print('Python', sys.version)
    print('input dirs', list(pathlib.Path('/kaggle/input').glob('*')) if pathlib.Path('/kaggle/input').exists() else None)
    sh(['apt-get','update','-qq'])
    sh(['apt-get','install','-y','-qq','ffmpeg','git','git-lfs','libgl1','libglib2.0-0'])
    sh(['git','lfs','install'])
    # P100-compatible torch baseline.
    sh([sys.executable,'-m','pip','uninstall','-y','torch','torchaudio','torchvision','torchtext','torchdata'])
    sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','--force-reinstall','torch==2.5.1+cu121','torchvision==0.20.1+cu121','torchaudio==2.5.1+cu121','--index-url','https://download.pytorch.org/whl/cu121'])
    # Ditto README wants numpy 2.0.1; avoid Kaggle default churn.
    sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','--force-reinstall','numpy==2.0.1','pillow==11.0.0'])
    sh([sys.executable,'-m','pip','install','-q','--no-cache-dir',
        'librosa==0.10.2.post1','tqdm','filetype','imageio','opencv-python-headless==4.10.0.84',
        'scikit-image==0.25.0','cython','imageio-ffmpeg','colored','huggingface_hub','soundfile','onnxruntime-gpu','mediapipe'])
    import torch, numpy as np
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    manifest['cuda_available']=bool(torch.cuda.is_available())
    manifest['cuda_device_name']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try: print(out(['nvidia-smi'])[:2000])
    except Exception as e: print('nvidia failed', e)

    portrait=find_asset('portrait.jpg')
    audio=find_asset('audio.wav')
    audio20=WORK/'audio_20s.wav'
    sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(audio),'-t','20','-ar','16000','-ac','1',str(audio20)])
    manifest['audio_duration']=out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(audio20)]).strip()

    os.chdir(WORK)
    sh(['git','clone','--depth','1','https://github.com/antgroup/ditto-talkinghead.git'])
    os.chdir(WORK/'ditto-talkinghead')
    # Download only PyTorch checkpoints + cfg from HF.
    from huggingface_hub import snapshot_download
    snapshot_download('digital-avatar/ditto-talkinghead', local_dir='checkpoints', local_dir_use_symlinks=False,
                      allow_patterns=['ditto_cfg/v0.4_hubert_cfg_pytorch.pkl','ditto_pytorch/**'])
    # Variant: reduce expression intensity while keeping head pose/translation.
    # Best observed strategy: keep natural head motion but damp mouth/expression deformation artifacts.
    import pickle
    more_kwargs = {
        'setup_kwargs': {
            'crop_scale': 2.45,
            'crop_vy_ratio': -0.10,
            'use_d_keys': {'exp': 0.65, 'pitch': 1.0, 'yaw': 1.0, 'roll': 1.0, 't': 1.0},
        },
        'run_kwargs': {}
    }
    kw_path = WORK/'ditto_more_kwargs_exp065.pkl'
    kw_path.write_bytes(pickle.dumps(more_kwargs))
    # Patch inference CLI to accept --more_kwargs and pass it into run().
    inf=pathlib.Path('inference.py')
    txt=inf.read_text()
    txt=txt.replace(
        'parser.add_argument("--output_path", type=str, help="path to output mp4")',
        'parser.add_argument("--output_path", type=str, help="path to output mp4")\n'
        '    parser.add_argument("--more_kwargs", type=str, default="", help="optional pickle with setup/run kwargs")'
    )
    txt=txt.replace("run(SDK, audio_path, source_path, output_path)", "run(SDK, audio_path, source_path, output_path, args.more_kwargs if args.more_kwargs else {})")
    inf.write_text(txt)
    out_raw=WORK/'result_ditto_raw_exp065.mp4'
    sh([sys.executable,'inference.py',
        '--data_root','./checkpoints/ditto_pytorch',
        '--cfg_pkl','./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl',
        '--audio_path',str(audio20),
        '--source_path',str(portrait),
        '--output_path',str(out_raw),
        '--more_kwargs',str(kw_path)])
    sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(out_raw),'-c:v','libx264','-pix_fmt','yuv420p','-profile:v','high','-level','4.1','-crf','23','-preset','veryfast','-c:a','aac','-b:a','128k','-movflags','+faststart',str(FINAL)])
    manifest.update({'output':FINAL.name,'duration_seconds':out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(FINAL)]).strip(),'bytes':FINAL.stat().st_size})
except Exception:
    manifest['errors']['pipeline']=traceback.format_exc()
    print(manifest['errors']['pipeline'])
    MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    raise
MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(MANIFEST.read_text())
# Cleanup heavy dirs.
for p in [WORK/'ditto-talkinghead', WORK/'result_ditto_raw.mp4', WORK/'audio_20s.wav']:
    if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
    elif p.exists(): p.unlink()
print('DONE', [p.name for p in WORK.iterdir() if p.is_file()])
