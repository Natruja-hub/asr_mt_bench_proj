# config.py
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR   = os.path.join(BASE_DIR, "data", "audio")
REF_ASR_DIR = os.path.join(BASE_DIR, "data", "references_asr")
REF_MT_DIR  = os.path.join(BASE_DIR, "data", "references_mt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000

# ─── ASR Models ──────────────────────────────────────────────────────────────
WHISPER_CONFIG = {
    "model_size": "small",   # ~460 MB
    "language":   "en",
    "fp16":       False,      # CPU mode
    "beam_size":  3,
}

DEEPSPEECH_CONFIG = {
    "model_path": "deepspeech-0.9.3-models.pbmm",
    "scorer_path": "deepspeech-0.9.3-models.scorer",
    "sample_rate": SAMPLE_RATE,
}


VOSK_CONFIG = {
    "model_path":  "vosk-model-en-us-0.22",  # ~1.8 GB — download: https://alphacephei.com/vosk/models
    "sample_rate": SAMPLE_RATE,
}

# ─── MT Models ───────────────────────────────────────────────────────────────
M2M100_CONFIG = {
    "model_name": "facebook/m2m100_418M",  
    "src_lang": "en",
    "tgt_lang": "th",
    "max_length": 512,
    "num_beams": 3,
}

LIBRETRANSLATE_CONFIG = {
    "url":     "http://localhost:5000/translate",
    "source":  "en",
    "target":  "th",
    "api_key": "",
    "timeout": 30,
}

NLLB_CONFIG = {
    "model_name": "facebook/nllb-200-distilled-600M",  # ~2.4 GB
    "src_lang":   "eng_Latn",
    "tgt_lang":   "tha_Thai",
    "max_length": 512,
    "num_beams":  3,
}

# ─── Pipeline combinations ───────────────────────────────────────────────────
ASR_MODELS = ["deepspeech","whisper", "vosk"]
MT_MODELS  = ["m2m100", "libretranslate", "nllb"]

# ─── Output files ────────────────────────────────────────────────────────────
RESULTS_CSV = os.path.join(RESULTS_DIR, "pipeline_results.csv")
ASR_CSV     = os.path.join(RESULTS_DIR, "asr_standalone.csv")
MT_CSV      = os.path.join(RESULTS_DIR, "mt_standalone.csv")
