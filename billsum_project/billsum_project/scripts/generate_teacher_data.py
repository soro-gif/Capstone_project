"""Utilise le modèle déjà entraîné (sorolamoussa/t5-small-billsum-fr) comme
teacher : génère un résumé pour chaque texte du dataset BillSum, et écrit
directement le JSONL au format chat OpenAI (system/user/assistant).

Usage:
    python scripts/generate_teacher_data.py --split train --out train_openai.jsonl
    python scripts/generate_teacher_data.py --split val --out val_openai.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from billsum.config import Config
from billsum.data import load_splits

TEACHER_MODEL = "sorolamoussa/t5-small-billsum-fr"
TEACHER_PREFIX = "summarize: "  # préfixe requis par ce modèle T5 (cf. audience.py)

SYSTEM_PROMPT = (
    "Tu es un assistant qui résume des projets de loi (bills) de façon "
    "précise et concise, en conservant les obligations légales importantes."
)


def to_chat_example(text: str, summary: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": summary},
        ]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--out", default="train_openai.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=12000,
                    help="tronque les textes trop longs avant tokenisation")
    p.add_argument("--limit", type=int, default=None,
                    help="limite le nombre de textes traités (utile pour tester)")
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement de {TEACHER_MODEL} sur {device}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL).to(device)
    model.eval()

    cfg = Config()
    train_ds, val_ds, _test_ds, _ca_test_ds = load_splits(cfg)
    dataset = train_ds if args.split == "train" else val_ds

    texts = [t[:args.max_chars] for t in dataset["text"] if t]
    if args.limit:
        texts = texts[:args.limit]
    print(f"{len(texts)} textes du split '{args.split}'")

    n_written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i:i + args.batch_size]
            enc = tokenizer(
                [TEACHER_PREFIX + t for t in chunk], max_length=1024, truncation=True, padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, num_beams=4,
                    no_repeat_ngram_size=5, repetition_penalty=1.15,
                    early_stopping=True,
                )
            summaries = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for text, summary in zip(chunk, summaries):
                if not summary.strip():
                    continue
                f.write(json.dumps(to_chat_example(text, summary), ensure_ascii=False) + "\n")
                n_written += 1
            print(f"{n_written}/{len(texts)}...")

    print(f"Écrit {n_written} exemples dans {args.out}")


if __name__ == "__main__":
    main()
