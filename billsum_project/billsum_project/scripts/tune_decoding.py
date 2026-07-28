"""Compare plusieurs jeux de paramètres de decoding sur un même texte, pour
choisir empiriquement les valeurs qui minimisent l'hallucination et
maximisent la couverture des obligations (pipeline FR de production).

Usage:
    python scripts/tune_decoding.py chemin/vers/texte.txt --audience CITOYEN
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from billsum.multipublic import summarize_for_audience

PARAM_GRID = [
    {"num_beams": 4, "repetition_penalty": 1.15, "no_repeat_ngram_size": 5, "length_penalty": 1.0},
    {"num_beams": 6, "repetition_penalty": 1.15, "no_repeat_ngram_size": 5, "length_penalty": 1.0},
    {"num_beams": 4, "repetition_penalty": 1.1, "no_repeat_ngram_size": 4, "length_penalty": 1.0},
    {"num_beams": 6, "repetition_penalty": 1.1, "no_repeat_ngram_size": 4, "length_penalty": 0.9},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text_file")
    p.add_argument("--audience", default="CITOYEN")
    args = p.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")

    print(f"{'params':60s} | oblig_cov | hallu_rate | len(mots)")
    print("-" * 100)
    for params in PARAM_GRID:
        result = summarize_for_audience(text, args.audience, **params)
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        n_words = len(result.summary.split())
        print(f"{label:60s} | {result.obligation_coverage:9.2f} | {result.hallucination_rate:10.2f} | {n_words}")


if __name__ == "__main__":
    main()
