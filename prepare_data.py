# prepare_data.py
# Run this once to copy audio files and create reference txt files
import os
import shutil
import csv

# Override this path with ASR_MT_CORPUS_DIR if the corpus is stored elsewhere.
CORPUS_DIR = os.environ.get(
    "ASR_MT_CORPUS_DIR",
    os.path.join(
        os.path.expanduser("~"),
        "Desktop",
        "sps-corpus-4.0-2026-06-12-en",
    ),
)
NUM_FILES  = 50   # how many files to prepare (start with 50)
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_SRC  = os.path.join(CORPUS_DIR, "audios")
TSV_FILE   = os.path.join(CORPUS_DIR, "ss-corpus-en.tsv")

AUDIO_DST  = os.path.join("data", "audio")
REF_ASR    = os.path.join("data", "references_asr")
REF_MT     = os.path.join("data", "references_mt")

os.makedirs(AUDIO_DST, exist_ok=True)
os.makedirs(REF_ASR,   exist_ok=True)
os.makedirs(REF_MT,    exist_ok=True)


def main():
    print(f"Reading {TSV_FILE} ...")

    # Read TSV and find sentence column
    with open(TSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [row for row in reader]

    print(f"Found {len(rows)} entries in TSV")
    print(f"Columns: {list(rows[0].keys())}")

    # Find the right columns — print first row so we can check
    print(f"\nFirst row sample: {rows[0]}\n")

    count = 0
    skipped = 0

    for row in rows:
        if count >= NUM_FILES:
            break

        # Common column names in Common Voice TSV
        # Try to find audio filename and sentence
        filename = (
            row.get("audio_file") or
            row.get("path") or
            row.get("filename") or
            row.get("audio") or
            row.get("file") or
            ""
        ).strip()

        sentence = (
            row.get("transcription") or
            row.get("sentence") or
            row.get("text") or
            row.get("transcript") or
            ""
        ).strip()

        if count == 0:
             print("filename:", repr(filename))
             print("sentence:", repr(sentence))

        if not filename or not sentence:
            skipped += 1
            continue

        # Add .mp3 if no extension
        if not os.path.splitext(filename)[1]:
            filename += ".mp3"

        src_audio = os.path.join(AUDIO_SRC, filename)
        if not os.path.exists(src_audio):
            # Try searching in subdirectories
            for root, dirs, files in os.walk(AUDIO_SRC):
                for f in files:
                    if f == filename or f == os.path.basename(filename):
                        src_audio = os.path.join(root, f)
                        break

        if not os.path.exists(src_audio):
            skipped += 1
            continue

        # Copy audio file
        base = os.path.splitext(os.path.basename(filename))[0]
        dst_audio = os.path.join(AUDIO_DST, os.path.basename(filename))
        shutil.copy2(src_audio, dst_audio)

        # Write ASR reference (English transcript)
        with open(os.path.join(REF_ASR, base + ".txt"), "w", encoding="utf-8") as f:
            f.write(sentence)

        # Write MT reference placeholder
        # (you need to fill these with Thai translations)
        mt_path = os.path.join(REF_MT, base + ".txt")
        if not os.path.exists(mt_path):
            with open(mt_path, "w", encoding="utf-8") as f:
                f.write(f"[TRANSLATE THIS]: {sentence}")

        count += 1
        print(f"[{count:>3}] {os.path.basename(filename)}")

    print(f"\n✓ Done! Copied {count} files, skipped {skipped}")
    print(f"\n⚠️  Next step: fill in Thai translations in data\\references_mt\\")
    print(f"   Open each .txt file and replace [TRANSLATE THIS]: ... with the Thai translation")
    print(f"   Or run: python translate_references.py  (if you want auto-translate)")


if __name__ == "__main__":
    main()