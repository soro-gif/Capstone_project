"""Génère un résumé exécutif (juriste) et un résumé citoyen d'un projet de loi FR,
avec traçabilité des sections couvertes/omises et vérification anti-hallucination.

Utilise le modèle pré-entraîné sorolamoussa/t5-small-billsum-fr
(https://huggingface.co/sorolamoussa/t5-small-billsum-fr) — pas d'entraînement requis.

Usage:
    python scripts/run_multipublic_fr.py --file projet_de_loi.txt
    python scripts/run_multipublic_fr.py --file projet_de_loi.txt --audiences JURISTE DIRIGEANT CITOYEN
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from billsum.multipublic import generate_report, FR_MODEL_NAME


def print_result(result):
    print(f"\n{'=' * 60}")
    print(f"RÉSUMÉ — public {result.audience}")
    print("=" * 60)
    print(result.summary)

    print("\n--- Traçabilité ---")
    print(f"Sections couvertes ({len(result.covered_sections)}): "
          f"{', '.join(result.covered_sections) or '(aucune)'}")
    if result.omitted_sections:
        print(f"[!] Sections OMISES ({len(result.omitted_sections)}): "
              f"{', '.join(result.omitted_sections)}")
    else:
        print("Aucune section omise détectée.")

    if result.obligation_coverage is not None:
        print(f"\nCouverture des obligations légales : {result.obligation_coverage * 100:.1f}%")
    else:
        print("\nAucune obligation légale détectée dans la source.")

    if result.hallucination_rate is not None:
        print(f"Taux d'hallucination (phrases non sourcées) : "
              f"{result.hallucination_rate * 100:.1f}%")
        if result.unsupported_sentences:
            print("[!] Phrases non sourcées par le texte original :")
            for s in result.unsupported_sentences:
                print(f"    - {s}")

    if result.glossary_used:
        print("\n--- Glossaire appliqué (registre citoyen) ---")
        for term, definition in result.glossary_used.items():
            print(f"  {term} -> {definition}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="fichier texte du projet de loi (FR)")
    p.add_argument("--audiences", nargs="+", default=["JURISTE", "CITOYEN"],
                   choices=["JURISTE", "DIRIGEANT", "CITOYEN"])
    p.add_argument("--model-name", default=FR_MODEL_NAME)
    args = p.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Modèle: {args.model_name}")

    report = generate_report(text, audiences=args.audiences, device=device,
                             model_name=args.model_name)
    for aud in args.audiences:
        print_result(report[aud])


if __name__ == "__main__":
    main()
