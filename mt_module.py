# mt_module.py
import requests
from monitor_module import ResourceMonitor
from config import M2M100_CONFIG, LIBRETRANSLATE_CONFIG, NLLB_CONFIG

_CACHE = {}


def _load_m2m100():
    if "M2M100" not in _CACHE:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        print("[MT] Loading M2M100...")

        name = M2M100_CONFIG["model_name"]

        tokenizer = M2M100Tokenizer.from_pretrained(name)
        tokenizer.src_lang = M2M100_CONFIG["src_lang"]

        _CACHE["M2M100"] = {
            "tokenizer": tokenizer,
            "model": M2M100ForConditionalGeneration.from_pretrained(name),
        }

    return _CACHE["M2M100"]

def _load_nllb():
    if "nllb" not in _CACHE:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        print("[MT] Loading NLLB-200...")

        name = NLLB_CONFIG["model_name"]

        tokenizer = AutoTokenizer.from_pretrained(
            name,
            src_lang=NLLB_CONFIG["src_lang"],
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(name)

        _CACHE["nllb"] = {
            "tokenizer": tokenizer,
            "model": model,
        }

    return _CACHE["nllb"]


def translate(text: str, model_name: str):
    model_name = model_name.lower()

    if model_name in ("m2m100", "m2m-100"):
        pkg = _load_m2m100()
        tokenizer, model = pkg["tokenizer"], pkg["model"]
        with ResourceMonitor() as m:
            inputs = tokenizer(
                text,
                return_tensors="pt", 
                truncation=True,
                max_length=M2M100_CONFIG["max_length"],
            )
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(
                M2M100_CONFIG["tgt_lang"]
                ),
                num_beams=M2M100_CONFIG["num_beams"],
                max_length=M2M100_CONFIG["max_length"],
            )
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation, m.report()

    elif model_name == "libretranslate":
        cfg = LIBRETRANSLATE_CONFIG
        payload = {"q": text, "source": cfg["source"], "target": cfg["target"], "api_key": cfg.get("api_key", "")}
        with ResourceMonitor() as m:
            response = requests.post(cfg["url"], json=payload, timeout=cfg["timeout"])
            response.raise_for_status()
            translation = response.json()["translatedText"]
        return translation, m.report()

    elif model_name == "nllb":
        pkg = _load_nllb()
        tokenizer, model = pkg["tokenizer"], pkg["model"]
        with ResourceMonitor() as m:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=NLLB_CONFIG["max_length"],
            )
            target_id = tokenizer.convert_tokens_to_ids(
               NLLB_CONFIG["tgt_lang"]
            )
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                num_beams=NLLB_CONFIG["num_beams"],
                max_length=NLLB_CONFIG["max_length"],
            )
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation, m.report()

    else:
        raise ValueError(f"Unknown MT model: {model_name!r}")
