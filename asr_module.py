# asr_module.py
import json
import numpy as np

from monitor_module import ResourceMonitor
from config import (
    SAMPLE_RATE,
    WHISPER_CONFIG,
    DEEPSPEECH_CONFIG,
    VOSK_CONFIG,
)

_CACHE = {}


def _load_whisper():
    if "whisper" not in _CACHE:
        import whisper

        print("[ASR] Loading Whisper...")

        _CACHE["whisper"] = whisper.load_model(
            WHISPER_CONFIG["model_size"]
        )

    return _CACHE["whisper"]


def _load_deepspeech():
    if "deepspeech" not in _CACHE:
        import deepspeech

        print("[ASR] Loading DeepSpeech...")

        model = deepspeech.Model(
            DEEPSPEECH_CONFIG["model_path"]
        )

        model.enableExternalScorer(
            DEEPSPEECH_CONFIG["scorer_path"]
        )

        _CACHE["deepspeech"] = model

    return _CACHE["deepspeech"]


def _load_vosk():
    if "vosk" not in _CACHE:
        from vosk import Model

        print("[ASR] Loading Vosk...")

        _CACHE["vosk"] = Model(
            VOSK_CONFIG["model_path"]
        )

    return _CACHE["vosk"]


def transcribe(audio: np.ndarray, duration: float, model_name: str):
    model_name = model_name.lower()

    # Whisper
    if model_name == "whisper":

        model = _load_whisper()

        with ResourceMonitor(duration) as m:
            result = model.transcribe(
                audio,
                language=WHISPER_CONFIG["language"],
                fp16=WHISPER_CONFIG["fp16"],
                beam_size=WHISPER_CONFIG["beam_size"],
            )

        return result["text"].strip(), m.report()

    # DeepSpeech
    elif model_name == "deepspeech":

        model = _load_deepspeech()

        # DeepSpeech ต้องการข้อมูลเสียงแบบ int16
        audio_int16 = (
            audio * 32768
        ).clip(-32768, 32767).astype(np.int16)

        with ResourceMonitor(duration) as m:
            text = model.stt(audio_int16)

        return text.strip(), m.report()

    # Vosk
    elif model_name == "vosk":

        from vosk import KaldiRecognizer

        model = _load_vosk()

        audio_int16 = (
            audio * 32768
        ).clip(-32768, 32767).astype(np.int16).tobytes()

        with ResourceMonitor(duration) as m:
            rec = KaldiRecognizer(
                model,
                VOSK_CONFIG["sample_rate"],
            )

            rec.AcceptWaveform(audio_int16)

            text = json.loads(
                rec.FinalResult()
            ).get("text", "").strip()

        return text, m.report()

    else:
        raise ValueError(
            f"Unknown ASR model: {model_name!r}"
        )