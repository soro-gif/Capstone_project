"""Construction du modèle seq2seq + adaptateurs LoRA."""
from .config import Config


def build_model(cfg: Config, tokenizer):
    from transformers import AutoModelForSeq2SeqLM
    from peft import LoraConfig, get_peft_model, TaskType

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.MODEL_NAME)
    if cfg.USE_CONTROL_TOKEN:
        model.resize_token_embeddings(len(tokenizer))

    name = cfg.MODEL_NAME.lower()
    target_modules = ["q", "v"] if "t5" in name else ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_lora(cfg: Config, adapter_dir: str, device: str):
    """Recharge un modèle de base + adaptateurs LoRA sauvegardés."""
    from transformers import AutoModelForSeq2SeqLM
    from peft import PeftModel

    base = AutoModelForSeq2SeqLM.from_pretrained(cfg.MODEL_NAME)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    return model


def summarize_fr_with_base_model(text_fr: str, cfg: Config, tokenizer, adapter_dir: str,
                                  device: str = "cpu", max_new_tokens: int = 256) -> str:
    """Résume un texte français avec le modèle anglais de base, via un pont de
    traduction FR->EN->EN (résumé)->EN->FR.

    À utiliser uniquement pour ce modèle de base entraîné en anglais ; le
    pipeline de production (`multipublic.py`) utilise un modèle déjà
    fine-tuné en français et n'a pas besoin de ce pont.
    """
    import torch
    from .translate import translate_fr_to_en, translate_en_to_fr

    text_en = translate_fr_to_en(text_fr, device=device)

    model = load_lora(cfg, adapter_dir, device)
    enc = tokenizer(text_en, max_length=cfg.MAX_SOURCE_LEN, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4)
    summary_en = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

    return translate_en_to_fr(summary_en, device=device)
