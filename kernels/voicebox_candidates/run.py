import json, os, pathlib, shutil, subprocess, sys, traceback
WORK=pathlib.Path('/kaggle/working')
OUT=WORK/'outputs'; OUT.mkdir(exist_ok=True)
MANIFEST={"candidates":[], "errors":{}}
TEXT_EN="Hello, this is a private-speaker voice cloning test generated on a Kaggle GPU. The goal is to preserve timbre and speaking style without committing private audio to git."
TEXT_ES="Hola, esta es una prueba de clonación de voz de un hablante privado generada en una GPU de Kaggle. El objetivo es preservar timbre y estilo sin subir audio privado a git."

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

def save_wav(path, audio, sr):
    import numpy as np, soundfile as sf
    audio=np.asarray(audio, dtype='float32').squeeze()
    m=max(1e-6, float(abs(audio).max()))
    if m>1: audio=audio/m*0.95
    sf.write(str(path), audio, sr)
    mp3=path.with_suffix('.mp3')
    sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(path),'-codec:a','libmp3lame','-b:a','128k',str(mp3)])
    return mp3

def record(name, wav, mp3, engine, lang, text):
    dur=out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(wav)]).strip()
    MANIFEST['candidates'].append({'name':name,'engine':engine,'language':lang,'wav':wav.name,'mp3':mp3.name,'duration':dur,'text':text})

print('Python', sys.version)
print('input dirs', list(pathlib.Path('/kaggle/input').glob('*')) if pathlib.Path('/kaggle/input').exists() else None)
sh(['apt-get','update','-qq'])
sh(['apt-get','install','-y','-qq','ffmpeg','libsm6','libxext6','libgl1','espeak-ng'])
# T4 is available; use stable torch/numpy stack from voicebox notes.
sh([sys.executable,'-m','pip','uninstall','-y','torch','torchaudio','torchvision','torchtext','torchdata'])
sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','--force-reinstall','torch==2.5.1+cu121','torchvision==0.20.1+cu121','torchaudio==2.5.1+cu121','--index-url','https://download.pytorch.org/whl/cu121'])
sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','--force-reinstall','numpy==1.26.4','pillow==11.3.0'])
sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','librosa>=0.10.0','soundfile>=0.12.0','numba>=0.60.0,<0.61.0','scipy','huggingface_hub','safetensors','transformers>=4.36.0,<=4.57.6','accelerate','tqdm'])
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
try: print(out(['nvidia-smi'])[:2000])
except Exception as e: print('nvidia failed', e)

ref_src=find_asset('reference_voice.wav')
ref=WORK/'ref_clean.wav'
sh(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(ref_src),'-ac','1','-ar','24000','-af','highpass=f=70,lowpass=f=9000,loudnorm=I=-18:TP=-2:LRA=11,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2',str(ref)])
print('ref duration', out(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nk=1:nw=1',str(ref)]).strip())

# Candidate 1: Chatterbox Turbo English (Voicebox engine)
try:
    sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','--no-deps','chatterbox-tts'])
    sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','conformer>=0.3.2','diffusers>=0.29.0','omegaconf','pykakasi','resemble-perth>=1.0.1','s3tokenizer','spacy-pkuseg','pyloudnorm'])
    from huggingface_hub import snapshot_download
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    model_dir=snapshot_download('ResembleAI/chatterbox-turbo', allow_patterns=['*.safetensors','*.json','*.txt','*.pt','*.model'])
    model=ChatterboxTurboTTS.from_local(model_dir, 'cuda')
    wav=model.generate(TEXT_EN, audio_prompt_path=str(ref), temperature=0.75, top_k=1000, top_p=0.95, repetition_penalty=1.2)
    import torch as _torch
    arr=wav.squeeze().detach().cpu().numpy() if isinstance(wav,_torch.Tensor) else wav
    sr=getattr(model,'sr',None) or getattr(model,'sample_rate',24000)
    path=OUT/'candidate_chatterbox_turbo_en.wav'; mp3=save_wav(path, arr, sr)
    record('candidate_chatterbox_turbo_en', path, mp3, 'Voicebox/Chatterbox Turbo', 'en', TEXT_EN)
    del model; torch.cuda.empty_cache()
except Exception:
    MANIFEST['errors']['chatterbox_turbo']=traceback.format_exc()
    print(MANIFEST['errors']['chatterbox_turbo'])

# Candidate 2: Chatterbox Multilingual Spanish
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model=ChatterboxMultilingualTTS.from_pretrained(device='cuda')
    wav=model.generate(TEXT_ES, language_id='es', audio_prompt_path=str(ref), exaggeration=0.45, cfg_weight=0.55, temperature=0.75, repetition_penalty=2.0)
    import torch as _torch
    arr=wav.squeeze().detach().cpu().numpy() if isinstance(wav,_torch.Tensor) else wav
    sr=getattr(model,'sr',None) or getattr(model,'sample_rate',24000)
    path=OUT/'candidate_chatterbox_multilingual_es.wav'; mp3=save_wav(path, arr, sr)
    record('candidate_chatterbox_multilingual_es', path, mp3, 'Voicebox/Chatterbox Multilingual', 'es', TEXT_ES)
    del model; torch.cuda.empty_cache()
except Exception:
    MANIFEST['errors']['chatterbox_multilingual']=traceback.format_exc()
    print(MANIFEST['errors']['chatterbox_multilingual'])

# Candidate 3: LuxTTS English
try:
    sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','--find-links','https://k2-fsa.github.io/icefall/piper_phonemize.html','piper-phonemize'])
    sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','git+https://github.com/ysharma3501/LinaCodec.git','git+https://github.com/ysharma3501/LuxTTS.git'])
    from zipvoice.luxvoice import LuxTTS
    model=LuxTTS(model_path='YatharthS/LuxTTS', device='cuda')
    enc=model.encode_prompt(prompt_audio=str(ref), duration=10, rms=0.01)
    wav=model.generate_speech(text=TEXT_EN, encode_dict=enc, num_steps=8, guidance_scale=3.0, t_shift=0.5, speed=1.0, return_smooth=False)
    arr=wav.detach().cpu().numpy().squeeze()
    path=OUT/'candidate_luxtts_en.wav'; mp3=save_wav(path, arr, 48000)
    record('candidate_luxtts_en', path, mp3, 'Voicebox/LuxTTS', 'en', TEXT_EN)
    del model; torch.cuda.empty_cache()
except Exception:
    MANIFEST['errors']['luxtts']=traceback.format_exc()
    print(MANIFEST['errors']['luxtts'])

# Candidate 4: Qwen3-TTS 0.6B voice clone (if package works)
try:
    sh([sys.executable,'-m','pip','install','-q','--no-cache-dir','qwen-tts>=0.0.5'])
    from qwen_tts import Qwen3TTSModel
    model=Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-Base', device_map='cuda', torch_dtype=torch.bfloat16)
    prompt=model.create_voice_clone_prompt(ref_audio=str(ref), ref_text='', x_vector_only_mode=False)
    wavs, sr=model.generate_voice_clone(text=TEXT_EN, voice_clone_prompt=prompt, language='English', instruct='Speak naturally, close to the reference speaker timbre.')
    path=OUT/'candidate_qwen_06b_en.wav'; mp3=save_wav(path, wavs[0], sr)
    record('candidate_qwen_06b_en', path, mp3, 'Voicebox/Qwen3-TTS 0.6B', 'en', TEXT_EN)
    del model; torch.cuda.empty_cache()
except Exception:
    MANIFEST['errors']['qwen_06b']=traceback.format_exc()
    print(MANIFEST['errors']['qwen_06b'])

(WORK/'manifest.json').write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(MANIFEST, indent=2, ensure_ascii=False))
# Remove heavy dirs/files except outputs + manifest
for p in [WORK/'ref_clean.wav']:
    if p.exists(): p.unlink()
print('DONE outputs:', [str(p) for p in OUT.glob('*')])
