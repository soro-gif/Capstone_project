"""Convertit un JSONL au format chat OpenAI (system/user/assistant) en JSON
au format attendu par le tuning Gemini (Google AI Studio) : une liste
d'exemples {"text_input": ..., "output": ...}.

Usage:
    python scripts/prepare_gemini_data.py --in train_openai.jsonl --out train_gemini.json
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    args = p.parse_args()

    examples = []
    with open(args.infile, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = {m["role"]: m["content"] for m in row["messages"]}
            examples.append({
                "text_input": msgs["user"],
                "output": msgs["assistant"],
            })

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Écrit {len(examples)} exemples dans {args.outfile}")


if __name__ == "__main__":
    main()
