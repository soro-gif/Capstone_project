"""Évaluation du modèle Gemini fine-tuné déployé sur Vertex AI.

Contrairement à scripts/run_eval.py (qui évalue le modèle local T5-LoRA),
ce script appelle l'endpoint Vertex réel (billsum.vertex_backend.generate_with_vertex)
et calcule les mêmes métriques (ROUGE, BERTScore, couverture des obligations,
taux d'hallucination), pour avoir une mesure objective avant/après toute
optimisation du fine-tuning.

Usage:
    python scripts/run_eval_vertex.py --n-eval 50
    python scripts/run_eval_vertex.py --n-eval 100 --public DIRIGEANT
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from billsum.config import Config
from billsum.data import load_splits
from billsum.metrics import eval_rouge_bert, mean_coverage, mean_hallucination
from billsum.vertex_backend import generate_with_vertex


def generate_predictions(sources, public, sleep_s):
    preds = []
    for i, text in enumerate(sources):
        preds.append(generate_with_vertex(text, public))
        if sleep_s:
            time.sleep(sleep_s)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(sources)} générés")
    return preds


def evaluate_split(name, sources, refs, public, device, sleep_s):
    print(f"[{name}] génération de {len(sources)} résumés via l'endpoint Vertex...")
    preds = generate_predictions(sources, public, sleep_s)

    print(f"[{name}] ROUGE + BERTScore...")
    rb = eval_rouge_bert(preds, refs)
    print(f"[{name}] couverture des obligations...")
    cov = mean_coverage(sources, preds, Config.COVERAGE_THRESHOLD, device)
    print(f"[{name}] hallucination rate...")
    hall = mean_hallucination(sources, preds, device)

    results = {**rb, "obligation_coverage": cov, "hallucination_rate": hall}
    print(f"\n===== RÉSULTATS [{name}] =====")
    for k, v in results.items():
        print(f"  {k:22s}: {v}")
    return results, preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--public", default="JURISTE", choices=["JURISTE", "DIRIGEANT", "CITOYEN"])
    p.add_argument("--sleep", type=float, default=0.0, help="pause (s) entre appels API pour éviter le rate limiting")
    p.add_argument("--skip-ca", action="store_true", help="ne pas évaluer ca_test (OOD)")
    args = p.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Config()
    _, _, test_ds, ca_test_ds = load_splits(cfg)

    n = min(args.n_eval, len(test_ds))
    test_subset = test_ds.select(range(n))
    res_congress, preds_congress = evaluate_split(
        "test (Congress)", list(test_subset["text"]), list(test_subset["summary"]),
        args.public, device, args.sleep,
    )

    rows = [res_congress]
    idx = ["test (Congress)"]

    if not args.skip_ca:
        n_ca = min(args.n_eval, len(ca_test_ds))
        ca_subset = ca_test_ds.select(range(n_ca))
        res_ca, preds_ca = evaluate_split(
            "ca_test (CA, OOD)", list(ca_subset["text"]), list(ca_subset["summary"]),
            args.public, device, args.sleep,
        )
        rows.append(res_ca)
        idx.append("ca_test (CA, OOD)")

    df = pd.DataFrame(rows, index=idx)
    print("\nSynthèse (le gap Congress -> CA mesure la généralisation) :")
    print(df.to_string())
    out_file = f"evaluation_results_vertex_{args.public.lower()}.csv"
    df.to_csv(out_file)
    print(f"\nSauvegardé dans {out_file}")


if __name__ == "__main__":
    main()
