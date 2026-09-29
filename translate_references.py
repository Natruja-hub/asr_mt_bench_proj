# translate_references.py
# Cleans disfluency tags from reference files then re-translates them
import os, re, time
from deep_translator import GoogleTranslator

REF_ASR = os.path.join("data", "references_asr")
REF_MT  = os.path.join("data", "references_mt")

translator = GoogleTranslator(source="en", target="th")

def clean(text):
    # Remove [disfluency], [noise], [laughter] etc.
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    return ' '.join(text.split()).strip()

files = sorted(f for f in os.listdir(REF_ASR) if f.endswith(".txt"))
print(f"Found {len(files)} files\n")

for i, fname in enumerate(files, 1):
    # Clean ASR reference first
    asr_path = os.path.join(REF_ASR, fname)
    with open(asr_path, encoding="utf-8") as f:
        raw = f.read().strip()
    cleaned = clean(raw)
    with open(asr_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # Re-translate MT reference from cleaned text
    mt_path = os.path.join(REF_MT, fname)
    try:
        thai = translator.translate(cleaned)
        with open(mt_path, "w", encoding="utf-8") as f:
            f.write(thai)
        print(f"[{i:>2}] ✓ {fname}")
        print(f"      EN: {cleaned[:70]}...")
        print(f"      TH: {thai[:70]}...\n")
        time.sleep(0.5)
    except Exception as e:
        print(f"[{i:>2}] ✗ ERROR {fname}: {e}")
        time.sleep(2)

print("✓ Done — re-run controller.py now")