"""Chargement des splits officiels BillSum et tokenisation longue."""
from .config import Config, PUBLIC_TOKENS


def build_prefix(cfg: Config, public: str | None = None) -> str:
    public = public or cfg.DEFAULT_PUBLIC
    if cfg.USE_CONTROL_TOKEN:
        return f"[PUBLIC={public}] summarize: "
    return "summarize: "


def load_splits(cfg: Config):
    """Retourne (train, val, test, ca_test). Sous-échantillonne le train seul."""
    from datasets import load_dataset

    try:
        raw = load_dataset("FiscalNote/billsum")
    except Exception:
        raw = load_dataset("billsum")
    train_full, test_ds, ca_test_ds = raw["train"], raw["test"], raw["ca_test"]

    if cfg.TRAIN_SUBSET and cfg.TRAIN_SUBSET < len(train_full):
        train_full = train_full.shuffle(seed=cfg.SEED).select(range(cfg.TRAIN_SUBSET))

    val_size = cfg.VAL_SUBSET if len(train_full) > cfg.VAL_SUBSET else max(1, int(len(train_full) * 0.1))
    split = train_full.train_test_split(test_size=val_size, seed=cfg.SEED)
    return split["train"], split["test"], test_ds, ca_test_ds


def setup_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<pad>"
    if cfg.USE_CONTROL_TOKEN:
        tok.add_special_tokens({"additional_special_tokens": PUBLIC_TOKENS})
    return tok


def make_preprocessor(cfg: Config, tokenizer):
    prefix = build_prefix(cfg)

    def preprocess(batch):
        inputs = [prefix + (t or "") for t in batch["text"]]
        model_inputs = tokenizer(
            inputs, max_length=cfg.MAX_SOURCE_LEN, truncation=True, padding=False
        )
        labels = tokenizer(
            text_target=batch["summary"],
            max_length=cfg.MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def tokenize_datasets(cfg: Config, tokenizer, train_ds, val_ds):
    preprocess = make_preprocessor(cfg, tokenizer)
    cols = train_ds.column_names
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=cols, desc="tok train")
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=cols, desc="tok val")
    return train_tok, val_tok
