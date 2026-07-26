"""Distillation multi-publics (FR) via Claude Opus 4.8, pour préparer un jeu
d'entraînement à partir d'un texte de loi français découpé en articles.

Contrairement à distill_multipublic.py (dataset BillSum EN), ce script part
d'un fichier texte français (ex: Constitution ivoirienne) et génère, pour
chaque article, un résumé JURISTE (issu de l'article lui-même, référence) et
des variantes DIRIGEANT/CITOYEN via l'API Claude.

Prérequis : export ANTHROPIC_API_KEY=...

Usage:
    python scripts/distill_fr_articles.py --file constitution_ci_2016.txt --out distilled_fr.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anthropic

from billsum.sections import split_sections

MODEL = "claude-opus-4-8"

DISTILL_PROMPTS = {
    "JURISTE": (
        "Résume cet article de loi pour un juriste. Conserve la précision "
        "terminologique et les obligations légales exactes (délais, montants, "
        "sanctions), en 2-3 phrases maximum, sans paraphrase approximative."
    ),
    "DIRIGEANT": (
        "Résume cet article de loi pour un dirigeant de PME ou d'organisation. "
        "Mets en avant : obligations concrètes, échéances, sanctions, impact "
        "organisationnel. Style direct et actionnable, 2-3 phrases maximum."
    ),
    "CITOYEN": (
        "Résume cet article de loi en langage clair pour un citoyen. Explique "
        "l'impact concret sur la vie quotidienne, sans jargon juridique, "
        "2-3 phrases maximum."
    ),
}


def distill_one(client, article_text, public):
    prompt = f"{DISTILL_PROMPTS[public]}\n\n---\n{article_text[:4000]}"
    msg = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="fichier texte de loi (FR)")
    p.add_argument("--out", default="distilled_fr.jsonl")
    p.add_argument("--min-chars", type=int, default=80,
                    help="ignorer les sections trop courtes (titres/chapitres vides)")
    args = p.parse_args()

    client = anthropic.Anthropic()
    text = Path(args.file).read_text(encoding="utf-8")
    sections = split_sections(text)

    n_written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for header, body in sections:
            if len(body) < args.min_chars:
                continue
            for public in ("JURISTE", "DIRIGEANT", "CITOYEN"):
                try:
                    summ = distill_one(client, body, public)
                    f.write(json.dumps(
                        {"text": body, "public": public, "summary": summ},
                        ensure_ascii=False) + "\n")
                    n_written += 1
                except Exception as e:  # noqa: BLE001
                    print(f"skip ({header}, {public}):", e)

    print(f"Écrit {n_written} exemples dans {args.out}")


if __name__ == "__main__":
    main()
