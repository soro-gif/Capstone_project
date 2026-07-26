"""Hallucination rate — chaque phrase du résumé doit être entailed par la source.

Taux = fraction de phrases NON impliquées (neutral/contradiction) par la source.
"""
NLI_MODEL_EN = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_MODEL_FR = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # multilingue, couvre le FR
# labels : 0=entailment, 1=neutral, 2=contradiction

_nli_tok = None
_nli_model = None
_nli_name = None


def _load_nli(device="cpu", lang="en"):
    global _nli_tok, _nli_model, _nli_name
    name = NLI_MODEL_FR if lang == "fr" else NLI_MODEL_EN
    if _nli_model is None or _nli_name != name:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from ..hf import retry_closed_hf_client

        _nli_tok, _nli_model = retry_closed_hf_client(
            lambda: (
                AutoTokenizer.from_pretrained(name),
                AutoModelForSequenceClassification.from_pretrained(name).to(device).eval(),
            )
        )
        _nli_name = name
    return _nli_tok, _nli_model


def _sentences(text):
    import nltk

    return [s.strip() for s in nltk.sent_tokenize(text or "") if s.strip()]


def _chunk_source(source, max_chars=2000):
    """Fenêtre la source (contexte NLI limité)."""
    sents = _sentences(source)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) > max_chars:
            if cur:
                chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks or [source[:max_chars]]


def _is_entailed(premise_chunks, hypothesis, device="cpu", lang="en"):
    import torch

    tok, model = _load_nli(device, lang=lang)
    pairs = [(c, hypothesis) for c in premise_chunks]
    enc = tok(
        [p for p, _ in pairs], [h for _, h in pairs],
        truncation=True, padding=True, max_length=512, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    labels = logits.argmax(dim=1)
    return (labels == 0).any().item()


def hallucination_rate(source, summary, device="cpu", lang="en", return_details=False):
    hyps = _sentences(summary)
    if not hyps:
        return (None, []) if return_details else None
    chunks = _chunk_source(source)
    flags = [_is_entailed(chunks, h, device, lang=lang) for h in hyps]
    non_entailed = sum(0 if f else 1 for f in flags)
    rate = non_entailed / len(hyps)
    if not return_details:
        return rate
    unsupported = [h for h, f in zip(hyps, flags) if not f]
    return rate, unsupported


def mean_hallucination(sources, summaries, device="cpu", lang="en"):
    vals = [hallucination_rate(s, p, device, lang=lang) for s, p in zip(sources, summaries)]
    vals = [v for v in vals if v is not None]
    return round(100 * sum(vals) / len(vals), 2) if vals else None
