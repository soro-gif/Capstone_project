"""Génération de résumés avec beam search."""
from .config import Config
from .data import build_prefix


def summarize_texts(cfg: Config, model, tokenizer, texts, public=None,
                    batch_size=4, max_new=None):
    import torch

    max_new = max_new or cfg.MAX_TARGET_LEN
    model.eval()
    prefix = build_prefix(cfg, public)
    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = [prefix + (t or "") for t in texts[i:i + batch_size]]
        enc = tokenizer(
            chunk, max_length=cfg.MAX_SOURCE_LEN, truncation=True,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new, num_beams=4,
                no_repeat_ngram_size=5, repetition_penalty=1.15,
                early_stopping=True, length_penalty=1.0,
            )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs
