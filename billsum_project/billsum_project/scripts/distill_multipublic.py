"""Génération de résumés multi-publics par distillation (Phase 2).

BillSum ne fournit qu'un résumé par loi. Ce script génère les variantes
DIRIGEANT et CITOYEN avec un modèle enseignant (API Claude), pour ensuite
réentraîner avec USE_CONTROL_TOKEN=True.

Prérequis : export ANTHROPIC_API_KEY=...
    pip install anthropic

Usage:
    python scripts/distill_multipublic.py --n 2000 --out distilled.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from billsum.config import Config
from billsum.data import load_splits

DISTILL_PROMPTS = {
    "DIRIGEANT": (
        "Résume ce texte de loi pour un dirigeant de PME. Mets en avant : "
        "obligations concrètes, échéances, sanctions, coûts. Style direct et actionnable."
    ),
    "CITOYEN": (
        "Résume ce texte de loi en langage clair pour un citoyen. Explique "
        "l'impact concret sur la vie quotidienne, sans jargon juridique."
    ),
}
MODEL = "claude-sonnet-4-6"


def distill_one(client, text, public):
    prompt = f"{DISTILL_PROMPTS[public]}\n\n---\n{text[:12000]}"
    msg = client.messages.create(
        model=MODEL, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", default="distilled.jsonl")
    args = p.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Définir ANTHROPIC_API_KEY d'abord.")
    import anthropic

    client = anthropic.Anthropic()
    cfg = Config()
    train_ds, _, _, _ = load_splits(cfg)
    train_ds = train_ds.select(range(min(args.n, len(train_ds))))

    with open(args.out, "w", encoding="utf-8") as f:
        # Le résumé officiel sert de registre JURISTE
        for ex in train_ds:
            f.write(json.dumps(
                {"text": ex["text"], "public": "JURISTE", "summary": ex["summary"]},
                ensure_ascii=False) + "\n")
            for public in ("DIRIGEANT", "CITOYEN"):
                try:
                    summ = distill_one(client, ex["text"], public)
                    f.write(json.dumps(
                        {"text": ex["text"], "public": public, "summary": summ},
                        ensure_ascii=False) + "\n")
                except Exception as e:  # noqa: BLE001
                    print("skip:", e)
    print("Écrit dans", args.out)


if __name__ == "__main__":
    main()
