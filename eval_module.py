# eval_module.py
from jiwer import wer, cer
from sacrebleu.metrics import BLEU, CHRF

_BLEU = BLEU(effective_order=True)
_CHRF = CHRF()


def eval_asr(hypothesis: str, reference: str) -> dict:
    if not hypothesis or not reference:
        return {"wer": 1.0, "cer": 1.0}
    return {
        "wer": round(wer(reference, hypothesis), 4),
        "cer": round(cer(reference, hypothesis), 4),
    }


def eval_mt(hypothesis: str, reference: str) -> dict:
    if not hypothesis or not reference:
        return {"bleu": 0.0, "chrf": 0.0}
    return {
        "bleu": round(_BLEU.sentence_score(hypothesis, [reference]).score, 4),
        "chrf": round(_CHRF.sentence_score(hypothesis, [reference]).score, 4),
    }


def eval_pipeline(asr_hyp, asr_ref, mt_hyp, mt_ref) -> dict:
    return {**eval_asr(asr_hyp, asr_ref), **eval_mt(mt_hyp, mt_ref)}
