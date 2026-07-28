"""Traduction FR<->EN pour utiliser le modèle de base anglais avec des entrées françaises.

Utilise les modèles MarianMT Helsinki-NLP (transformers + sentencepiece, déjà
dépendances du projet, aucun nouveau package requis).
"""

MODEL_FR_EN = "Helsinki-NLP/opus-mt-fr-en"
MODEL_EN_FR = "Helsinki-NLP/opus-mt-en-fr"

_cache = {}


def _load(model_name: str, device: str = "cpu"):
    if model_name not in _cache:
        from transformers import MarianMTModel, MarianTokenizer

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device).eval()
        _cache[model_name] = (model, tokenizer)
    return _cache[model_name]


def _translate(text: str, model_name: str, device: str = "cpu", max_length: int = 512) -> str:
    import torch

    model, tokenizer = _load(model_name, device)
    chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)] if len(text) > 2000 else [text]

    translated = []
    for chunk in chunks:
        enc = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=max_length, num_beams=4)
        translated.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
    return " ".join(translated)


def translate_fr_to_en(text: str, device: str = "cpu") -> str:
    return _translate(text, MODEL_FR_EN, device)


def translate_en_to_fr(text: str, device: str = "cpu") -> str:
    return _translate(text, MODEL_EN_FR, device)
