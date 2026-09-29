# controller.py — runs pipeline 3 times, saves all runs + average + summary
import os, csv, statistics
from collections import defaultdict
from tqdm import tqdm

from config       import ASR_MODELS, MT_MODELS, RESULTS_CSV, ASR_CSV, MT_CSV
from input_module import load_audio, load_reference_asr, load_reference_mt, list_audio_files
from asr_module   import transcribe
from mt_module    import translate
from eval_module  import eval_asr, eval_mt, eval_pipeline

NUM_RUNS = 3

# ── CSV helper ────────────────────────────────────────────────────────────────
def _write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# ── Average helper ────────────────────────────────────────────────────────────
METRIC_COLS = ["wer", "cer", "bleu", "chrf",
               "asr_latency", "mt_latency", "total_latency",
               "total_ram_mb", "total_vram_mb", "asr_rtf"]

def _average_rows(rows, group_key, label_col="pipeline"):
    """Group rows by group_key and return one average row per group."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)
    avg_rows = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        avg = {label_col: key, "run": "AVERAGE", "file": f"(n={n})"}
        for col in METRIC_COLS:
            vals = [float(r[col]) for r in rs if r.get(col) not in (None, "")]
            avg[col] = round(statistics.mean(vals), 4) if vals else ""
        avg_rows.append(avg)
    return avg_rows

# ── Summary helper ────────────────────────────────────────────────────────────
def _summary_rows(pipeline_rows):
    """One summary row per pipeline averaged across all runs and files."""
    groups = defaultdict(list)
    for r in pipeline_rows:
        groups[r["pipeline"]].append(r)

    summary = []
    for pipeline, rs in sorted(groups.items()):
        row = {"pipeline": pipeline, "n_samples": len(rs)}
        for col in METRIC_COLS:
            vals = [float(r[col]) for r in rs if r.get(col) not in (None, "")]
            row[f"avg_{col}"] = round(statistics.mean(vals), 4) if vals else ""
            row[f"std_{col}"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        summary.append(row)
    return summary

# ── STEP 1: Standalone ASR ────────────────────────────────────────────────────
def run_asr_standalone(audio_files):
    print("\n" + "="*60)
    print("STEP 1: Standalone ASR Test")
    print("="*60)
    rows = []
    for asr in ASR_MODELS:
        print(f"\n[ASR] {asr}")
        for run in range(1, NUM_RUNS + 1):
            for fp in tqdm(audio_files, desc=f"{asr} run {run}/{NUM_RUNS}"):
                audio, dur = load_audio(fp)
                ref_asr    = load_reference_asr(fp)
                hyp, m     = transcribe(audio, dur, asr)
                scores     = eval_asr(hyp, ref_asr)
                rows.append({
                    "asr_model":      asr,
                    "run":            run,
                    "file":           os.path.basename(fp),
                    "hypothesis":     hyp,
                    "reference":      ref_asr,
                    **scores,
                    "asr_latency":    m["latency_sec"],
                    "asr_rtf":        m["rtf"],
                    "total_ram_mb":   m["ram_used_mb"],
                    "total_vram_mb":  m["vram_used_mb"],
                })

    # Add average rows per ASR model
    avg_rows = []
    for asr in ASR_MODELS:
        model_rows = [r for r in rows if r["asr_model"] == asr]
        n = len(model_rows)
        avg = {"asr_model": asr, "run": "AVERAGE", "file": f"(n={n})",
               "hypothesis": "", "reference": ""}
        for col in ["wer", "cer", "asr_latency", "asr_rtf", "total_ram_mb", "total_vram_mb"]:
            vals = [float(r[col]) for r in model_rows if r.get(col) not in (None, "")]
            avg[col] = round(statistics.mean(vals), 4) if vals else ""
        avg_rows.append(avg)

    _write_csv(ASR_CSV, rows + avg_rows)
    print(f"\n[✓] {ASR_CSV}")
    return rows

# ── STEP 2: Standalone MT ─────────────────────────────────────────────────────
def run_mt_standalone(audio_files):
    print("\n" + "="*60)
    print("STEP 2: Standalone MT Test")
    print("="*60)

    print("Transcribing with Whisper as input source...")
    src = {}
    for fp in tqdm(audio_files, desc="whisper baseline"):
        audio, dur = load_audio(fp)
        text, _    = transcribe(audio, dur, "whisper")
        src[fp]    = text

    rows = []
    for mt in MT_MODELS:
        print(f"\n[MT] {mt}")
        for run in range(1, NUM_RUNS + 1):
            for fp in tqdm(audio_files, desc=f"{mt} run {run}/{NUM_RUNS}"):
                ref_mt           = load_reference_mt(fp)
                translation, m   = translate(src[fp], mt)
                scores           = eval_mt(translation, ref_mt)
                rows.append({
                    "mt_model":      mt,
                    "run":           run,
                    "file":          os.path.basename(fp),
                    "source_text":   src[fp],
                    "translation":   translation,
                    "reference":     ref_mt,
                    **scores,
                    "mt_latency":    m["latency_sec"],
                    "total_ram_mb":  m["ram_used_mb"],
                    "total_vram_mb": m["vram_used_mb"],
                })

    # Add average rows per MT model
    avg_rows = []
    for mt in MT_MODELS:
        model_rows = [r for r in rows if r["mt_model"] == mt]
        n = len(model_rows)
        avg = {"mt_model": mt, "run": "AVERAGE", "file": f"(n={n})",
               "source_text": "", "translation": "", "reference": ""}
        for col in ["bleu", "chrf", "mt_latency", "total_ram_mb", "total_vram_mb"]:
            vals = [float(r[col]) for r in model_rows if r.get(col) not in (None, "")]
            avg[col] = round(statistics.mean(vals), 4) if vals else ""
        avg_rows.append(avg)

    _write_csv(MT_CSV, rows + avg_rows)
    print(f"\n[✓] {MT_CSV}")
    return rows

# ── STEP 3: Pipeline 3×3 ─────────────────────────────────────────────────────
def run_pipeline_3x3(audio_files):
    print("\n" + "="*60)
    print("STEP 3: Integrated 3×3 Pipeline Test (3 runs each)")
    print("="*60)
    rows  = []
    total = len(ASR_MODELS) * len(MT_MODELS) * len(audio_files) * NUM_RUNS

    with tqdm(total=total, desc="Pipeline 3×3") as pbar:
        for asr in ASR_MODELS:
            for mt in MT_MODELS:
                pipeline_id = f"{asr}+{mt}"
                for run in range(1, NUM_RUNS + 1):
                    for fp in audio_files:
                        audio, dur = load_audio(fp)
                        ref_asr    = load_reference_asr(fp)
                        ref_mt     = load_reference_mt(fp)
                        asr_text, am = transcribe(audio, dur, asr)
                        translation, mm = translate(asr_text, mt)
                        scores = eval_pipeline(asr_text, ref_asr, translation, ref_mt)
                        rows.append({
                            "pipeline":      pipeline_id,
                            "asr_model":     asr,
                            "mt_model":      mt,
                            "run":           run,
                            "file":          os.path.basename(fp),
                            "asr_text":      asr_text,
                            "translation":   translation,
                            **scores,
                            "asr_latency":   am["latency_sec"],
                            "asr_rtf":       am["rtf"],
                            "mt_latency":    mm["latency_sec"],
                            "total_latency": round(am["latency_sec"] + mm["latency_sec"], 4),
                            "total_ram_mb":  round(am["ram_used_mb"]  + mm["ram_used_mb"],  2),
                            "total_vram_mb": round(am["vram_used_mb"] + mm["vram_used_mb"], 2),
                        })
                        pbar.update(1)

    # Average rows per pipeline per run
    avg_rows = []
    for asr in ASR_MODELS:
        for mt in MT_MODELS:
            pipeline_id = f"{asr}+{mt}"
            for run in range(1, NUM_RUNS + 1):
                run_rows = [r for r in rows if r["pipeline"] == pipeline_id and r["run"] == run]
                n = len(run_rows)
                avg = {"pipeline": pipeline_id, "asr_model": asr, "mt_model": mt,
                       "run": f"run{run}_AVG", "file": f"(n={n})",
                       "asr_text": "", "translation": ""}
                for col in METRIC_COLS:
                    vals = [float(r[col]) for r in run_rows if r.get(col) not in (None, "")]
                    avg[col] = round(statistics.mean(vals), 4) if vals else ""
                avg_rows.append(avg)

    # Overall average per pipeline (all runs)
    overall_rows = []
    for asr in ASR_MODELS:
        for mt in MT_MODELS:
            pipeline_id = f"{asr}+{mt}"
            pipeline_rows = [r for r in rows if r["pipeline"] == pipeline_id]
            n = len(pipeline_rows)
            avg = {"pipeline": pipeline_id, "asr_model": asr, "mt_model": mt,
                   "run": "OVERALL_AVG", "file": f"(n={n}, {NUM_RUNS} runs)",
                   "asr_text": "", "translation": ""}
            for col in METRIC_COLS:
                vals = [float(r[col]) for r in pipeline_rows if r.get(col) not in (None, "")]
                avg[col] = round(statistics.mean(vals), 4) if vals else ""
            overall_rows.append(avg)

    _write_csv(RESULTS_CSV, rows + avg_rows + overall_rows)
    print(f"\n[✓] {RESULTS_CSV}")
    return rows

# ── SUMMARY ───────────────────────────────────────────────────────────────────
def print_and_save_summary(pipeline_rows):
    print("\n" + "="*60)
    print("SUMMARY — All 9 Pipelines (average across 5 runs)")
    print("="*60)

    summary = _summary_rows(pipeline_rows)

    cols = ["wer", "cer", "bleu", "chrf", "total_latency", "total_ram_mb"]
    header = f"{'Pipeline':<32}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['pipeline']:<32}" + "".join(f"{row.get('avg_'+c, ''):>14}" for c in cols))

    # Save summary CSV
    summary_path = RESULTS_CSV.replace("pipeline_results.csv", "summary.csv")
    _write_csv(summary_path, summary)
    print(f"\n[✓] Summary saved → {summary_path}")
    return summary

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    audio_files = list_audio_files()
    print(f"[INFO] Found {len(audio_files)} audio files — running {NUM_RUNS} times each")
    print(f"[INFO] Total pipeline runs: {len(ASR_MODELS)} × {len(MT_MODELS)} × {len(audio_files)} × {NUM_RUNS} = "
          f"{len(ASR_MODELS)*len(MT_MODELS)*len(audio_files)*NUM_RUNS}")

    run_asr_standalone(audio_files)
    run_mt_standalone(audio_files)
    pipeline_rows = run_pipeline_3x3(audio_files)
    print_and_save_summary(pipeline_rows)

    print("\n[✓] Done — results saved to results/")
    print("  pipeline_results.csv — all runs + per-run averages + overall averages")
    print("  asr_standalone.csv   — all ASR runs + averages")
    print("  mt_standalone.csv    — all MT runs + averages")
    print("  summary.csv          — one row per pipeline with avg ± std")