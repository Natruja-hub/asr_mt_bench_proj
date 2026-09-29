# input_module.py
import os
import librosa
import numpy as np
from config import AUDIO_DIR, REF_ASR_DIR, REF_MT_DIR, SAMPLE_RATE


def load_audio(filepath: str, target_sr: int = SAMPLE_RATE):
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    duration = len(audio) / target_sr
    return audio.astype(np.float32), duration


def load_reference_asr(audio_filename: str) -> str:
    base = os.path.splitext(os.path.basename(audio_filename))[0]
    ref_path = os.path.join(REF_ASR_DIR, base + ".txt")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"ASR reference not found: {ref_path}")
    with open(ref_path, encoding="utf-8") as f:
        return f.read().strip()


def load_reference_mt(audio_filename: str) -> str:
    base = os.path.splitext(os.path.basename(audio_filename))[0]
    ref_path = os.path.join(REF_MT_DIR, base + ".txt")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"MT reference not found: {ref_path}")
    with open(ref_path, encoding="utf-8") as f:
        return f.read().strip()


def list_audio_files(directory: str = AUDIO_DIR):
    exts = {".wav", ".mp3", ".ogg", ".flac"}
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        raise FileNotFoundError(f"No audio files found in {directory}")
    return files
