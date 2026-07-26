"""Convertit un JSONL au format chat OpenAI (system/user/assistant) en JSONL
au format attendu par le tuning supervisé Vertex AI :
{"systemInstruction": {...}, "contents": [{"role": "user", "parts": [...]}, {"role": "model", "parts": [...]}]}

Usage:
    python scripts/prepare_vertex_data.py --in train_openai.jsonl --out train_vertex.jsonl
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    args = p.parse_args()

    n = 0
    with open(args.infile, encoding="utf-8") as fin, \
         open(args.outfile, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            msgs = {m["role"]: m["content"] for m in row["messages"]}
            example = {
                "systemInstruction": {
                    "role": "system",
                    "parts": [{"text": msgs["system"]}],
                },
                "contents": [
                    {"role": "user", "parts": [{"text": msgs["user"]}]},
                    {"role": "model", "parts": [{"text": msgs["assistant"]}]},
                ],
            }
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            n += 1

    print(f"Écrit {n} exemples dans {args.outfile}")


if __name__ == "__main__":
    main()
