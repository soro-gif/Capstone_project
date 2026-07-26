"""Lance un job de tuning Gemini via Google AI Studio (SDK google-genai).

Prérequis : export GEMINI_API_KEY=... (clé obtenue sur https://aistudio.google.com/apikey)
Usage:
    python scripts/run_gemini_finetune.py --train train_gemini.json --name billsum-fr --watch
"""
import argparse
import json
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="JSON produit par prepare_gemini_data.py")
    p.add_argument("--base-model", default="models/gemini-1.5-flash-001-tuning")
    p.add_argument("--name", default="billsum-fr")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--watch", action="store_true", help="suit le job jusqu'à la fin")
    args = p.parse_args()

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Définissez GEMINI_API_KEY (ou GOOGLE_API_KEY) avant de lancer.")
    client = genai.Client(api_key=api_key)

    with open(args.train, encoding="utf-8") as f:
        raw_examples = json.load(f)
    print(f"{len(raw_examples)} exemples chargés depuis {args.train}")

    examples = [
        types.TuningExample(text_input=ex["text_input"], output=ex["output"])
        for ex in raw_examples
    ]
    training_dataset = types.TuningDataset(examples=examples)

    config = types.CreateTuningJobConfig(
        tuned_model_display_name=args.name,
        epoch_count=args.epochs,
    )

    job = client.tunings.tune(
        base_model=args.base_model,
        training_dataset=training_dataset,
        config=config,
    )
    print(f"Job de tuning créé : {job.name} (state={job.state})")

    if args.watch:
        while job.state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            time.sleep(20)
            job = client.tunings.get(name=job.name)
            print(f"state={job.state}")
        if job.state == "JOB_STATE_SUCCEEDED":
            print(f"Modèle tuné : {job.tuned_model.model}")
        else:
            print("Échec ou annulation :", job.error)
    else:
        print(f"Suivi manuel : client.tunings.get(name='{job.name}')")


if __name__ == "__main__":
    main()
