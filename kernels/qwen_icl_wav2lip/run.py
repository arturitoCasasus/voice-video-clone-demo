import json, os, pathlib, shutil, subprocess, sys, traceback
WORK=pathlib.Path('/kaggle/working')
FINAL=WORK/'result_qwen_06b_icl_wav2lip.mp4'
MANIFEST=WORK/'manifest.json'
TARGET_TEXT="Hello, this is a Qwen ICL voice cloning test using a clean private reference recording and exact reference text. The goal is to evaluate whether ICL mode improves speaker similarity."

def sh(cmd, **kw):
    print('\n$', cmd if isinstance(cmd,str) else ' '.join(map(str,cmd)), flush=True)
    return subprocess.check_call(cmd, **kw)

def out(cmd): return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def find_asset(name):
    roots=[pathlib.Path('/kaggle/input'), pathlib.Path('/kaggle/working'), pathlib.Path('/kaggle/src')]
    hits=[]
    for r in roots:
        if r.exists(): hits.extend(r.rglob(name))
    if not hits: raise FileNotFoundError(name)
    print('asset', name, '->', hits[0])
    return hits[0]

manifest={'target_text':TARGET_TEXT,'engine':'Qwen3-TTS 0.6B Base ICL clone + Wav2Lip','errors':{}}
print('Python', sys.version)
print('input dirs', list(pathlib.Path('/kaggle/input').glob('*')) if pathlib.Path('/kaggle/input').exists() else None)
sh(['apt-get','update','-qq'])
sh(['apt-get','install','-y','-qq','ffmpeg','libsm6','libxext6','libgl1','wget','git','espeak-ng'])
sh([sys.executable,'-m','pip','uninstall','-y','torch','torchaudio','torchvision','torchtext','torchdata'])
sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','--force-reinstall','torch==2.5.1+cu121','torchvision==0.20.1+cu121','torchaudio==2.5.1+cu121','--index-url','https://download.pytorch.org/whl/cu121'])
sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','--force-reinstall','numpy==1.26.4','pillow==11.3.0'])
sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','qwen-tts>=0.0.5','soundfile','librosa>=0.10.0','scipy','huggingface_hub','safetensors','transformers>=4.36.0,<=4.57.6','accelerate','tqdm','opencv-python','face-alignment','pydub'])

import torch, soundfile as sf, numpy as np
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
try: print(out(['nvidia-smi'])[:2000])
except Exception as e: print('nvidia failed', e)
manifest['cuda_available']=bool(torch.cuda.is_available())
manifest['cuda_device_name']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

portrait=find_asset('portrait.jpg')
ref=find_asset('reference_qwen_icl.wav')
ref_text=find_asset('reference_text.txt').read_text(encoding='utf-8').strip()
manifest['reference_text']=ref_text
manifest['reference_text_chars']=len(ref_text)
manifest['reference_duration']=out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(ref)]).strip()

voice_wav=WORK/'qwen_06b_icl_voice.wav'
try:
    from qwen_tts import Qwen3TTSModel
    model=Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cuda' if torch.cuda.is_available() else 'cpu', torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
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
        max_new_tokens=2048,
    )
    arr=np.asarray(wavs[0], dtype='float32').squeeze()
    m=max(1e-6, float(np.abs(arr).max()))
    if m>1: arr=arr/m*0.95
    sf.write(str(voice_wav), arr, sr)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
except Exception:
    manifest['errors']['qwen_06b_icl']=traceback.format_exc()
    print(manifest['errors']['qwen_06b_icl'])
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    raise

voice_16k=WORK/'qwen_06b_icl_voice_16k.wav'
sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(voice_wav),'-ar','16000','-ac','1','-af','loudnorm=I=-18:TP=-2:LRA=11',str(voice_16k)])
manifest['voice_duration']=out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(voice_16k)]).strip()

# Wav2Lip on CPU to avoid P100 CUDA issues.
os.chdir(WORK)
sh(['git','clone','--depth','1','https://github.com/Rudrabha/Wav2Lip.git'])
ckpt=WORK/'Wav2Lip/checkpoints/wav2lip_gan.pth'; ckpt.parent.mkdir(parents=True, exist_ok=True)
for url in ['https://huggingface.co/rippertnt/wav2lip/resolve/main/wav2lip_gan.pth?download=true','https://huggingface.co/EraSpire/wav2lip/resolve/main/wav2lip_gan.pth?download=true','https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth']:
    try:
        sh(['wget','-q','-O',str(ckpt),url])
        if ckpt.exists() and ckpt.stat().st_size>100_000_000: break
    except Exception as e: print('checkpoint download failed', url, repr(e))
if not ckpt.exists() or ckpt.stat().st_size<100_000_000: raise RuntimeError('checkpoint download failed')
ap=WORK/'Wav2Lip/audio.py'
s=ap.read_text()
s=s.replace('librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,', 'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,')
s=s.replace('librosa.filters.mel(hp.sample_rate, hp.n_fft, hp.num_mels, hp.fmin, hp.fmax)', 'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)')
ap.write_text(s)
os.chdir(WORK/'Wav2Lip')
raw=WORK/'result_raw.mp4'
cmd=[sys.executable,'inference.py','--checkpoint_path',str(ckpt),'--face',str(portrait),'--audio',str(voice_16k),'--outfile',str(raw),'--static','True','--fps','25','--pads','0','20','0','0','--resize_factor','3','--face_det_batch_size','1','--wav2lip_batch_size','4']
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=''
sh(cmd, env=env)
sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(raw),'-vf',"drawtext=text='AI-generated demo':fontcolor=white:fontsize=22:box=1:boxcolor=black@0.45:x=16:y=16,format=yuv420p",'-c:v','libx264','-crf','28','-preset','veryfast','-c:a','aac','-b:a','96k','-movflags','+faststart',str(FINAL)])
manifest.update({'output':FINAL.name,'duration_seconds':out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(FINAL)]).strip(),'bytes':FINAL.stat().st_size})
MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(MANIFEST.read_text())
os.chdir(WORK)
for target in ['Wav2Lip','result_raw.mp4','qwen_06b_icl_voice_16k.wav']:
    p=WORK/target
    if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
    elif p.exists(): p.unlink()
print('DONE', [p.name for p in WORK.iterdir() if p.is_file()])
