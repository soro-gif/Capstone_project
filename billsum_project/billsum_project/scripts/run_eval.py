"""Évaluation complète sur test (Congress) et ca_test (Californie, hors-domaine).

Usage:
    python scripts/run_eval.py --adapter ./billsum-lora
    python scripts/run_eval.py --adapter ./billsum-lora --n-eval 200
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import pandas as pd

from billsum.config import Config
from billsum.data import load_splits, setup_tokenizer
from billsum.model import load_lora
from billsum.metrics import full_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None, help="dossier des adaptateurs LoRA")
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--model-name", default=None)
    args = p.parse_args()

    cfg = Config()
    if args.model_name:
        cfg.MODEL_NAME = args.model_name
    if args.n_eval is not None:
        cfg.N_EVAL = args.n_eval
    adapter = args.adapter or cfg.OUTPUT_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| Adaptateur:", adapter)

    _, _, test_ds, ca_test_ds = load_splits(cfg)
    tokenizer = setup_tokenizer(cfg)
    model = load_lora(cfg, adapter, device)

    res_congress = full_evaluation(cfg, model, tokenizer, test_ds,
                                   "test (Congress)", device)
    res_ca = full_evaluation(cfg, model, tokenizer, ca_test_ds,
                             "ca_test (CA, OOD)", device)

    df = pd.DataFrame([res_congress, res_ca],
                      index=["test (Congress)", "ca_test (CA, OOD)"])
    print("\nSynthèse (le gap Congress -> CA mesure la généralisation) :")
    print(df.to_string())
    df.to_csv("evaluation_results.csv")
    print("\nSauvegardé dans evaluation_results.csv")


if __name__ == "__main__":
    main()
