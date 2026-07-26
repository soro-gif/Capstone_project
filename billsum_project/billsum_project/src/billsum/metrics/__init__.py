"""Évaluation : ROUGE-L, BERTScore, couverture des obligations, hallucination."""
from .obligations import mean_coverage, obligation_coverage, extract_obligations
from .hallucination import mean_hallucination, hallucination_rate

__all__ = [
    "eval_rouge_bert",
    "mean_coverage",
    "obligation_coverage",
    "extract_obligations",
    "mean_hallucination",
    "hallucination_rate",
    "full_evaluation",
]


def eval_rouge_bert(preds, refs):
    import evaluate as hf_evaluate
    from bert_score import score as bertscore_fn

    rouge = hf_evaluate.load("rouge")
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    _, _, F1 = bertscore_fn(preds, refs, lang="en", rescale_with_baseline=True)
    return {
        "rougeL": round(r["rougeL"] * 100, 2),
        "rouge1": round(r["rouge1"] * 100, 2),
        "rouge2": round(r["rouge2"] * 100, 2),
        "bertscore_f1": round(F1.mean().item() * 100, 2),
    }


def full_evaluation(cfg, model, tokenizer, dataset, name, device,
                    n_eval=None, public=None):
    from ..generate import summarize_texts

    n_eval = n_eval or cfg.N_EVAL
    subset = dataset.select(range(min(n_eval, len(dataset))))
    sources, refs = subset["text"], subset["summary"]

    print(f"[{name}] génération de {len(sources)} résumés...")
    preds = summarize_texts(cfg, model, tokenizer, sources, public=public)

    print(f"[{name}] ROUGE + BERTScore...")
    rb = eval_rouge_bert(preds, refs)
    print(f"[{name}] couverture des obligations...")
    cov = mean_coverage(sources, preds, cfg.COVERAGE_THRESHOLD, device)
    print(f"[{name}] hallucination rate...")
    hall = mean_hallucination(sources, preds, device)

    results = {**rb, "obligation_coverage": cov, "hallucination_rate": hall}
    print(f"\n===== RÉSULTATS [{name}] =====")
    for k, v in results.items():
        print(f"  {k:22s}: {v}")
    return results
