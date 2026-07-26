"""Entraînement LoRA de bout en bout.

Usage:
    python scripts/run_train.py
    python scripts/run_train.py --train-subset 15000 --epochs 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from billsum.config import Config
from billsum.data import load_splits, setup_tokenizer, tokenize_datasets
from billsum.model import build_model
from billsum.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default=None)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--control-token", action="store_true", help="active le multi-publics")
    args = p.parse_args()

    cfg = Config()
    if args.model_name:
        cfg.MODEL_NAME = args.model_name
    if args.train_subset is not None:
        cfg.TRAIN_SUBSET = args.train_subset
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.control_token:
        cfg.USE_CONTROL_TOKEN = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| Modèle:", cfg.MODEL_NAME)

    train_ds, val_ds, _, _ = load_splits(cfg)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    tokenizer = setup_tokenizer(cfg)
    model = build_model(cfg, tokenizer)
    train_tok, val_tok = tokenize_datasets(cfg, tokenizer, train_ds, val_ds)

    train(cfg, model, tokenizer, train_tok, val_tok, device)
    print("Terminé. Adaptateurs LoRA dans", cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
