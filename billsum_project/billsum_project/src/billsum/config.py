"""Configuration centralisée du projet BillSum multi-publics."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- Modèle ---
    MODEL_NAME: str = "google/long-t5-tglobal-base"  # contexte long natif
    # MODEL_NAME = "allenai/led-base-16384"          # alternative LED (16k)
    # MODEL_NAME = "facebook/bart-large-cnn"         # alternative légère (1024)

    # --- Longueurs ---
    MAX_SOURCE_LEN: int = 4096  # baisser à 2048 si VRAM limitée
    MAX_TARGET_LEN: int = 512

    # --- Échantillonnage (splits officiels conservés) ---
    TRAIN_SUBSET: int = 12000  # None = tout ; 10-15k conseillé
    VAL_SUBSET: int = 500

    # --- LoRA ---
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.1

    # --- Entraînement ---
    EPOCHS: int = 3
    LR: float = 3e-4
    BATCH_SIZE: int = 2
    GRAD_ACCUM: int = 8  # batch effectif = 16
    WEIGHT_DECAY: float = 0.01
    LABEL_SMOOTHING: float = 0.1  # réduit l'overconfidence -> moins d'hallucinations
    WARMUP_RATIO: float = 0.05

    # --- Multi-publics ---
    USE_CONTROL_TOKEN: bool = False
    DEFAULT_PUBLIC: str = "JURISTE"

    # --- Évaluation ---
    N_EVAL: int = 100
    COVERAGE_THRESHOLD: float = 0.55

    OUTPUT_DIR: str = "./billsum-lora"
    SEED: int = 42


PUBLIC_TOKENS = ["[PUBLIC=JURISTE]", "[PUBLIC=DIRIGEANT]", "[PUBLIC=CITOYEN]"]
