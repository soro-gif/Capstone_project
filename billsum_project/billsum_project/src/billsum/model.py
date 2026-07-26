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
