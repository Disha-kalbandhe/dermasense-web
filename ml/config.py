import json
import os
from pathlib import Path

import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATASET_DIR_0 = DATA_RAW / "DATASET" / "DATASET_0" / "DATASET_0"
DATASET_DIR_1 = DATA_RAW / "DATASET" / "DATASET_1" / "DATASET_1"
DATASET_DIR = DATA_RAW / "DATASET"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
HEATMAPS_DIR = OUTPUTS_DIR / "gradcam_samples"
REPORTS_DIR = OUTPUTS_DIR / "reports"
HIERARCHY_MAP_PATH = DATA_PROCESSED / "hierarchy_map.json"

# ── CSV Paths ──────────────────────────────────────────────────────────────────
TRAIN_CSV = DATA_RAW / "train_split.csv"
TEST_CSV = DATA_RAW / "test_split.csv"
METADATA_CSV = DATA_RAW / "Skin_Metadata.csv"

# ── Column Names (confirmed from EDA) ─────────────────────────────────────────
LABEL_COL      = "Disease_label"
MAIN_CLASS_COL = "Main_class"
IMG_COL        = "Image_name"
FITZPATRICK_COL= "Fitzpatrick"
AGE_COL        = "Age"
SEX_COL        = "Sex"

# Tabular feature groups
BODY_PART_PREFIX  = "Body_part_"
DESCRIPTOR_PREFIX = "Descriptor_"
N_TABULAR_FEATS   = 96   # 49 body + 47 descriptor

# ── Model ──────────────────────────────────────────────────────────────────────
BACKBONE     = "efficientnet_b3"
IMG_SIZE     = 224
IN_CHANNELS  = 3
TABULAR_DIM  = 96           # input to tabular MLP branch
FUSION_DIM   = 256          # concat fusion hidden size
DROPOUT      = 0.3
D_MODEL      = 256
N_ATTN_LAYERS = 2
N_HEADS      = 8
LOSS_WEIGHTS = {"mainclass": 0.2, "subclass": 0.3, "disease": 1.0}
FOCAL_GAMMA  = 2.0
CB_BETA      = 0.999
NUM_MAIN_CLASSES = 8
NUM_SUBCLASS_CLASSES = 19

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 16
NUM_EPOCHS    = 8
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
SCHEDULER     = "cosine"
WARMUP_EPOCHS = 5
SEED          = 42

# Loss function
LABEL_SMOOTHING = 0.1   # already set implicitly — add explicitly
USE_CLASS_WEIGHTS = True
GRADIENT_ACCUMULATION_STEPS = 2
PATIENCE = 7
MIN_DELTA = 0.001

# ── Augmentation ───────────────────────────────────────────────────────────────
TRAIN_AUG = True
VAL_AUG   = False

# ── Hardware ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
PROFILE = "gpu" if torch.cuda.is_available() else "cpu_fast"
CPU_THREADS = min(os.cpu_count() or 1, 16)
NUM_CLASSES = len(json.loads((DATA_PROCESSED / "label2idx.json").read_text(encoding="utf-8")))

# ── Validation Split ───────────────────────────────────────────────────────────
VAL_SPLIT = 0.15

# ── Checkpoints ───────────────────────────────────────────────────────────────
SAVE_BEST_ONLY = True
BEST_MODEL = CHECKPOINTS_DIR / "best_model.pth"
LAST_MODEL = CHECKPOINTS_DIR / "last_model.pth"