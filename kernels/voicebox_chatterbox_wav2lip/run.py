import json, pathlib, shutil, subprocess, sys, os
os.environ['CUDA_VISIBLE_DEVICES']=''  # Force CPU Wav2Lip to avoid Kaggle P100/PyTorch CUDA incompatibility
WORK=pathlib.Path('/kaggle/working')
SRC=pathlib.Path('/kaggle/input')
FINAL=WORK/'result_voicebox_chatterbox_wav2lip.mp4'
MANIFEST=WORK/'manifest.json'
TEXT="Hello, this is a private-speaker voice cloning test generated on a Kaggle GPU. The goal is to preserve timbre and speaking style without committing private audio to git."

def sh(cmd, **kw):
    print('\n$', cmd if isinstance(cmd,str) else ' '.join(map(str,cmd)), flush=True)
    return subprocess.check_call(cmd, **kw)

def out(cmd): return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def find_file(name):
    roots=[pathlib.Path('/kaggle/input'), pathlib.Path('/kaggle/working'), pathlib.Path('/kaggle/src')]
    hits=[]
    for r in roots:
        if r.exists(): hits.extend(r.rglob(name))
    if not hits: raise FileNotFoundError(name)
    print('asset', name, '->', hits[0])
    return hits[0]

print('Python', sys.version)
print('input dirs', list(pathlib.Path('/kaggle/input').glob('*')) if pathlib.Path('/kaggle/input').exists() else None)
sh(['apt-get','update','-qq'])
sh(['apt-get','install','-y','-qq','ffmpeg','libsm6','libxext6','libgl1','wget','git'])
sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','opencv-python','librosa>=0.10.0','numpy<2','tqdm','face-alignment','scipy','pillow'])

portrait=find_file('portrait.jpg')
audio=find_file('voicebox_chatterbox_turbo_en.wav')
voice_16k=WORK/'voice_16k.wav'
sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(audio),'-ar','16000','-ac','1','-af','loudnorm=I=-18:TP=-2:LRA=11',str(voice_16k)])
print('voice duration', out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(voice_16k)]).strip())

os.chdir(WORK)
sh(['git','clone','--depth','1','https://github.com/Rudrabha/Wav2Lip.git'])
ckpt=WORK/'Wav2Lip/checkpoints/wav2lip_gan.pth'
ckpt.parent.mkdir(parents=True, exist_ok=True)
urls=[
 'https://huggingface.co/rippertnt/wav2lip/resolve/main/wav2lip_gan.pth?download=true',
 'https://huggingface.co/EraSpire/wav2lip/resolve/main/wav2lip_gan.pth?download=true',
 'https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth'
]
ok=False
for url in urls:
    try:
        sh(['wget','-q','-O',str(ckpt),url])
        ok=ckpt.exists() and ckpt.stat().st_size>100_000_000
        print('ckpt size', ckpt.stat().st_size if ckpt.exists() else 0)
        if ok: break
    except Exception as e:
        print('download failed', url, repr(e))
if not ok: raise RuntimeError('checkpoint download failed')

ap=WORK/'Wav2Lip/audio.py'
s=ap.read_text()
s=s.replace('librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,', 'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,')
s=s.replace('librosa.filters.mel(hp.sample_rate, hp.n_fft, hp.num_mels, hp.fmin, hp.fmax)', 'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)')
ap.write_text(s)

os.chdir(WORK/'Wav2Lip')
raw=WORK/'result_raw.mp4'
cmd=[sys.executable,'inference.py','--checkpoint_path',str(ckpt),'--face',str(portrait),'--audio',str(voice_16k),'--outfile',str(raw),'--static','True','--fps','25','--pads','0','20','0','0','--resize_factor','3','--face_det_batch_size','1','--wav2lip_batch_size','8']
sh(cmd)
sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(raw),'-vf',"drawtext=text='AI-generated demo':fontcolor=white:fontsize=22:box=1:boxcolor=black@0.45:x=16:y=16,format=yuv420p",'-c:v','libx264','-crf','28','-preset','veryfast','-c:a','aac','-b:a','96k','-movflags','+faststart',str(FINAL)])
manifest={'output':FINAL.name,'engine':'Voicebox/Chatterbox Turbo audio candidate + Wav2Lip','text':TEXT,'duration_seconds':out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(FINAL)]).strip(),'bytes':FINAL.stat().st_size}
MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(MANIFEST.read_text())
# leave final + manifest only
os.chdir(WORK)
for target in ['Wav2Lip','result_raw.mp4','voice_16k.wav']:
    p=WORK/target
    if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
    elif p.exists(): p.unlink()
print('DONE', [p.name for p in WORK.iterdir() if p.is_file()])
