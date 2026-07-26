"""Entraînement LoRA avec régularisation et early stopping sur ROUGE-L."""
import numpy as np

from .config import Config


def make_compute_metrics(tokenizer):
    import evaluate as hf_evaluate

    rouge = hf_evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dp = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
        res = rouge.compute(predictions=dp, references=dl, use_stemmer=True)
        return {k: round(v * 100, 2) for k, v in res.items()}

    return compute_metrics


def train(cfg: Config, model, tokenizer, train_tok, val_tok, device: str):
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        weight_decay=cfg.WEIGHT_DECAY,
        label_smoothing_factor=cfg.LABEL_SMOOTHING,
        warmup_ratio=cfg.WARMUP_RATIO,
        predict_with_generate=True,
        generation_max_length=cfg.MAX_TARGET_LEN,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=(device == "cuda"),
        report_to="none",
        seed=cfg.SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    trainer.save_model(cfg.OUTPUT_DIR)
    tokenizer.save_pretrained(cfg.OUTPUT_DIR)
    return trainer
