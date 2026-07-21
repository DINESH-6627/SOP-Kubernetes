# =============================================================
#  config.py
#  Central configuration — all hyperparameters, paths, settings.
#
#  PRIMARY ENVIRONMENT : local  (VS Code on Mac — default)
#
#  TO SWITCH ENVIRONMENT:
#    Change ENVIRONMENT variable below to "kaggle" or "cluster"
#    Everything else adjusts automatically.
#
#  Rule: this is the ONLY file you ever edit to change settings.
# =============================================================

import os


def _find_existing_path(*candidates: str) -> str:
    """Return first existing path, else first candidate as fallback."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]

# -------------------------------------------------------------
# 0. ENVIRONMENT SELECTOR  ← only line you need to change
# -------------------------------------------------------------
ENVIRONMENT = "local"    # "local" | "kaggle" | "cluster"

# -------------------------------------------------------------
# 1. PATHS
# -------------------------------------------------------------

if ENVIRONMENT == "local":
    # ── VS Code on Mac ────────────────────────────────────────
    # Prefer repository root derived from this file location.
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Support both layouts:
    #  1) <project>/data/*.csv
    #  2) <project>/*.csv
    BENIGN_CSV = _find_existing_path(
        os.path.join(BASE_DIR, "data", "benign.csv"),
        os.path.join(BASE_DIR, "benign.csv"),
    )
    MALICIOUS_CSV = _find_existing_path(
        os.path.join(BASE_DIR, "data", "malicious.csv"),
        os.path.join(BASE_DIR, "malicious.csv"),
    )

elif ENVIRONMENT == "kaggle":
    # ── Kaggle Notebook ───────────────────────────────────────
    # Add dataset via Notebook → Add Data →
    #   "sever-and-dogan-2023-dataset" by dinesh6627
    BASE_DIR        = "/kaggle/working/kubernetes_anomaly_detection"
    KAGGLE_DATA_DIR = ("/kaggle/input/datasets/"
                       "dinesh6627/sever-and-dogan-2023-dataset")
    BENIGN_CSV      = os.path.join(KAGGLE_DATA_DIR, "benign.csv")
    MALICIOUS_CSV   = os.path.join(KAGGLE_DATA_DIR, "malicious.csv")

elif ENVIRONMENT == "cluster":
    # ── College GPU cluster ───────────────────────────────────
    # Update BASE_DIR to your actual home directory on cluster.
    # scp all scripts and data there before running.
    BASE_DIR      = "/home/<your_username>/kubernetes_anomaly_detection"
    BENIGN_CSV    = os.path.join(BASE_DIR, "data", "benign.csv")
    MALICIOUS_CSV = os.path.join(BASE_DIR, "data", "malicious.csv")

else:
    raise ValueError(
        f"Unknown ENVIRONMENT: '{ENVIRONMENT}'. "
        f"Choose from: 'local', 'kaggle', 'cluster'"
    )

# ── Derived directories (same for all environments) ───────────
DATA_DIR       = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
SCRIPTS_DIR    = os.path.join(BASE_DIR, "scripts")
NOTEBOOKS_DIR  = os.path.join(BASE_DIR, "notebooks")

KAGGLE_DATASET = "dinesh6627/sever-and-dogan-2023-dataset"

# -------------------------------------------------------------
# 2. DEVICE
# -------------------------------------------------------------
# Auto-detected in training scripts. Do not set manually.
# Apple Silicon Mac  → "mps"   (Metal GPU — confirmed working)
# NVIDIA GPU         → "cuda"
# CPU fallback       → "cpu"

# -------------------------------------------------------------
# 3. REPRODUCIBILITY
# -------------------------------------------------------------
RANDOM_SEED = 42

# -------------------------------------------------------------
# 4. DATASET AND CLASS LABELS
# -------------------------------------------------------------
NUM_CLASSES   = 11
LABEL_COLUMN  = "label"      # normalized — lowercase + underscore
TIMESTAMP_COL = "timestamp"  # flow start time — used for sorting

CLASS_NAMES = {
    0:  "Benign",
    1:  "CVE-2020-13379",
    2:  "Node-RED Recon",
    3:  "Node-RED RCE",
    4:  "Node-RED Escape",
    5:  "CVE-2021-43798",
    6:  "CVE-2019-20933",
    7:  "CVE-2021-30465",
    8:  "CVE-2021-25741",
    9:  "CVE-2022-23648",
    10: "CVE-2019-5736",
}

CLASS_SHORT = {
    0:  "Benign",
    1:  "CVE-13379",
    2:  "Recon",
    3:  "RCE",
    4:  "Escape",
    5:  "CVE-43798",
    6:  "CVE-20933",
    7:  "CVE-30465",
    8:  "CVE-25741",
    9:  "CVE-23648",
    10: "CVE-5736",
}

# -------------------------------------------------------------
# 5. PREPROCESSING
# -------------------------------------------------------------
NAN_DROP_THRESHOLD    = 0.20   # drop cols with >20% NaN or Inf
VARIANCE_THRESHOLD    = 0.001  # drop near-zero variance features
CORRELATION_THRESHOLD = 0.95   # drop one of highly correlated pair
# No PCA or autoencoders — deliberate design decision.
# Differentiates from Aly et al. [32] and Allahabadi [34].

# -------------------------------------------------------------
# 6. SEQUENCE CONSTRUCTION  (Novelty 1)
# -------------------------------------------------------------
WINDOW_SIZE = 16     # W — flows per sequence
STRIDE      = 8      # S — sliding window stride
TIE_BREAK   = "attack"   # tie-break in favour of attack class

# -------------------------------------------------------------
# 7. TRAIN / VAL / TEST SPLIT
# -------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# -------------------------------------------------------------
# 8. TRANSFORMER ARCHITECTURE  (Novelty 1)
# -------------------------------------------------------------
D_MODEL   = 128    # embedding dimension
N_HEADS   = 8      # attention heads
N_LAYERS  = 4      # encoder layers
D_FF      = 512    # feed-forward inner dimension
DROPOUT   = 0.1    # dropout rate
FC_HIDDEN = 128    # FC layer before output
# Add this to config.py after D_MODEL
INPUT_DIM = 39    # D — confirmed after Stage 2 preprocessing
                  # 87 original cols → 39 after cleaning
WINDOW_SIZE = 16    # W — confirmed after Stage 3 analysis

# -------------------------------------------------------------
# 9. TRAINING
# -------------------------------------------------------------
BATCH_SIZE     = 64
EPOCHS         = 100
LEARNING_RATE  = 3e-4
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 5
EARLY_STOP_PAT = 10
LOSS           = "weighted_ce"   # "weighted_ce" | "focal"
FOCAL_GAMMA    = 2.0

# -------------------------------------------------------------
# 10. BASELINE MODELS
# -------------------------------------------------------------
RF_N_ESTIMATORS  = 100
RF_MAX_DEPTH     = None
SVM_KERNEL       = "rbf"
SVM_C            = 1.0
MLP_HIDDEN_SIZES = (256, 128, 64)
MLP_MAX_ITER     = 200
LSTM_HIDDEN      = 128
LSTM_LAYERS      = 2
LSTM_DROPOUT     = 0.2

# -------------------------------------------------------------
# 11. ABLATION SEARCH SPACE
# -------------------------------------------------------------
ABLATION_WINDOWS = [16, 32, 64]
ABLATION_LAYERS  = [1, 2, 4, 6]
ABLATION_HEADS   = [1, 4, 8]
ABLATION_POS_ENC = [True, False]

# -------------------------------------------------------------
# 12. EVALUATION  (Novelty 3)
# -------------------------------------------------------------
PRIMARY_METRIC   = "macro_f1"
METRICS          = ["macro_f1", "weighted_f1", "accuracy", "roc_auc"]
CONFUSION_MATRIX = True
PER_CLASS_REPORT = True
SAVE_PLOTS       = True

# -------------------------------------------------------------
# 13. SANITY CHECK
# -------------------------------------------------------------
print(f"[config] Environment  : {ENVIRONMENT}")
print(f"[config] Base dir     : {BASE_DIR}")
print(f"[config] Benign CSV   : {BENIGN_CSV}")
print(f"[config] Malicious    : {MALICIOUS_CSV}")
print(f"[config] Window       : W={WINDOW_SIZE}, S={STRIDE}")
print(f"[config] Model        : d={D_MODEL}, heads={N_HEADS}, "
      f"layers={N_LAYERS}, ff={D_FF}")