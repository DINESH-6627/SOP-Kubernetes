# =============================================================
#  stage9_ablation.py
#  Stage 9 — Ablation Study
#
#  Systematically evaluates the contribution of each
#  architectural and hyperparameter choice by varying
#  one component at a time while keeping all others fixed.
#
#  Four ablation dimensions (paper Section 7):
#
#  7.1 Window size W
#      Tests W ∈ {16, 32, 64} — how much temporal context
#      is needed to detect multi-stage attack progressions.
#      Default: W=16 (chosen for rare class coverage).
#
#  7.2 Number of encoder layers N
#      Tests N ∈ {1, 2, 4, 6} — depth of the encoder stack.
#      Default: N=4.
#
#  7.3 Number of attention heads h
#      Tests h ∈ {1, 4, 8} — breadth of attention patterns.
#      Default: h=8.
#
#  7.4 Positional encoding
#      Tests {True, False} — does temporal position matter?
#      If removing PE hurts performance on sequential attack
#      classes, this validates that position is meaningful.
#      Default: True (sinusoidal).
#
#  Each ablation run is a full training loop with early
#  stopping. All other settings remain fixed at defaults.
#
#  RECOMMENDED ENVIRONMENT: College GPU cluster (CUDA).
#  Runtime on NVIDIA V100: ~2-4 hours for full suite.
#  Runtime on Mac MPS: ~8-12 hours (feasible but slow).
#
#  Input  : numpy arrays from stage3 (already saved)
#  Output : ablation_results.json and ablation figures
#           saved to results/
# =============================================================

import os
import sys
import importlib.util
import math
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings("ignore")

# =============================================================
#  DYNAMIC PROJECT ROOT & CONFIG LOADING
# =============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Add PROJECT_ROOT to sys.path for stage imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.py")
spec = importlib.util.spec_from_file_location(
    "project_config", CONFIG_PATH
)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load config module from: {CONFIG_PATH}")

project_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_config)

# Load all required config variables
DATA_DIR         = project_config.DATA_DIR
RESULTS_DIR      = project_config.RESULTS_DIR
CHECKPOINT_DIR   = project_config.CHECKPOINT_DIR
LOG_DIR          = project_config.LOG_DIR
NUM_CLASSES      = project_config.NUM_CLASSES
CLASS_NAMES      = project_config.CLASS_NAMES
BATCH_SIZE       = project_config.BATCH_SIZE
LEARNING_RATE    = project_config.LEARNING_RATE
WEIGHT_DECAY     = project_config.WEIGHT_DECAY
EPOCHS           = project_config.EPOCHS
WARMUP_EPOCHS    = project_config.WARMUP_EPOCHS
EARLY_STOP_PAT   = project_config.EARLY_STOP_PAT
RANDOM_SEED      = project_config.RANDOM_SEED
ABLATION_WINDOWS = project_config.ABLATION_WINDOWS
ABLATION_LAYERS  = project_config.ABLATION_LAYERS
ABLATION_HEADS   = project_config.ABLATION_HEADS
ABLATION_POS_ENC = project_config.ABLATION_POS_ENC
D_MODEL          = project_config.D_MODEL
N_HEADS          = project_config.N_HEADS
N_LAYERS         = project_config.N_LAYERS
D_FF             = project_config.D_FF
DROPOUT          = project_config.DROPOUT
FC_HIDDEN        = project_config.FC_HIDDEN
WINDOW_SIZE      = project_config.WINDOW_SIZE
STRIDE           = project_config.STRIDE

INPUT_DIM = 39

# =============================================================
#  ABLATION RESULTS DIRECTORY
# =============================================================

ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation")
os.makedirs(ABLATION_DIR, exist_ok=True)

# =============================================================
#  1. DEVICE DETECTION
# =============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] CUDA GPU → {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[device] Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print(f"[device] CPU (slow — recommend GPU)")
    return device

# =============================================================
#  2. MODIFIED TRANSFORMER FOR ABLATION
# =============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal PE — identical to stage5_model.py."""
    def __init__(self, d_model, max_len=512, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AblationTransformer(nn.Module):
    """
    Transformer with configurable ablation parameters.
    Identical to KubernetesTransformer in stage5 but accepts
    n_layers, n_heads, and use_pos_enc as arguments for ablation.
    """
    def __init__(self,
                 input_dim:    int   = INPUT_DIM,
                 d_model:      int   = D_MODEL,
                 n_heads:      int   = N_HEADS,
                 n_layers:     int   = N_LAYERS,
                 d_ff:         int   = D_FF,
                 dropout:      float = DROPOUT,
                 fc_hidden:    int   = FC_HIDDEN,
                 n_classes:    int   = NUM_CLASSES,
                 use_pos_enc:  bool  = True):
        super().__init__()
        self.use_pos_enc = use_pos_enc

        self.input_projection = nn.Linear(input_dim, d_model)

        if use_pos_enc:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model, dropout=dropout
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # pre-norm
        )
        self.encoder      = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_projection(x)
        if self.use_pos_enc:
            x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

# =============================================================
#  3. LR SCHEDULER
# =============================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs,
                 total_epochs, base_lr, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = ((epoch - self.warmup_epochs) /
                        max(1, self.total_epochs -
                            self.warmup_epochs))
            lr = self.min_lr + 0.5 * (
                self.base_lr - self.min_lr
            ) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

# =============================================================
#  4. SEQUENCE BUILDER FOR WINDOW ABLATION
# =============================================================

def build_sequences_for_window(df_values: np.ndarray,
                                df_labels: np.ndarray,
                                window_size: int,
                                stride: int = STRIDE) -> tuple:
    """
    Build sequences with a specific window size.
    Used for window size ablation (W ∈ {16, 32, 64}).
    """
    from collections import Counter

    X_list, y_list = [], []
    n_flows = len(df_values)

    i = 0
    while i + window_size <= n_flows:
        window_X = df_values[i: i + window_size]
        window_y = df_labels[i: i + window_size]
        counts   = Counter(window_y.tolist())
        max_cnt  = max(counts.values())
        candidates = [l for l, c in counts.items()
                      if c == max_cnt]
        if len(candidates) == 1:
            label = int(candidates[0])
        else:
            attack = [c for c in candidates if c > 0]
            label  = int(max(attack)) if attack \
                else int(candidates[0])
        X_list.append(window_X)
        y_list.append(label)
        i += stride

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.int64))


def rebuild_splits_for_window(window_size: int) -> tuple:
    """
    Rebuild train/val/test splits for a given window size.
    Loads preprocessed.csv and re-runs sequence construction.
    Required for window size ablation.
    """
    import pandas as pd
    import pickle
    from collections import defaultdict

    pre_path  = os.path.join(DATA_DIR, "preprocessed.csv")
    feat_path = os.path.join(DATA_DIR, "feature_names.pkl")

    df = pd.read_csv(pre_path, low_memory=False)
    with open(feat_path, "rb") as f:
        feature_cols = pickle.load(f)

    values = df[feature_cols].values
    labels = df["label"].values

    X, y = build_sequences_for_window(values, labels,
                                       window_size, STRIDE)

    # Minimum-guarantee split (same as stage3)
    rng = np.random.RandomState(RANDOM_SEED)
    class_indices = defaultdict(list)
    for idx, lbl in enumerate(y):
        class_indices[int(lbl)].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    THRESH = 10

    for cls in sorted(class_indices.keys()):
        idxs     = class_indices[cls]
        n        = len(idxs)
        shuffled = rng.permutation(idxs).tolist()

        if n <= THRESH:
            val_idx.append(shuffled[0])
            if n >= 2:
                test_idx.append(shuffled[1])
                train_idx.extend(shuffled[2:])
            else:
                test_idx.append(shuffled[0])
        else:
            n_val   = max(1, int(round(n * 0.15)))
            n_test  = max(1, int(round(n * 0.15)))
            n_train = n - n_val - n_test
            val_idx.extend(shuffled[:n_val])
            test_idx.extend(shuffled[n_val:n_val+n_test])
            train_idx.extend(shuffled[n_val+n_test:])

    perm    = rng.permutation(len(train_idx))
    tr_arr  = np.array(train_idx)[perm]

    X_train, y_train = X[tr_arr], y[tr_arr]
    X_val,   y_val   = X[val_idx],  y[val_idx]
    X_test,  y_test  = X[test_idx], y[test_idx]

    # Re-fit scaler on train only
    from sklearn.preprocessing import StandardScaler
    D       = X_train.shape[2]
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, D)).reshape(X_train.shape
    ).astype(np.float32)
    X_val   = scaler.transform(
        X_val.reshape(-1, D)).reshape(X_val.shape
    ).astype(np.float32)
    X_test  = scaler.transform(
        X_test.reshape(-1, D)).reshape(X_test.shape
    ).astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test

# =============================================================
#  5. SINGLE ABLATION TRAINING RUN
# =============================================================

def run_single_ablation(X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val:   np.ndarray,
                         y_val:   np.ndarray,
                         X_test:  np.ndarray,
                         y_test:  np.ndarray,
                         device:  torch.device,
                         n_layers:    int   = N_LAYERS,
                         n_heads:     int   = N_HEADS,
                         use_pos_enc: bool  = True,
                         run_name:    str   = "ablation") -> dict:
    """
    Train one ablation configuration and return test metrics.
    Uses same training protocol as stage6 for fair comparison.
    """
    from scripts.stage4_dataloader import KubernetesSequenceDataset

    train_ds = KubernetesSequenceDataset(X_train, y_train)
    val_ds   = KubernetesSequenceDataset(X_val,   y_val)
    test_ds  = KubernetesSequenceDataset(X_test,  y_test)

    # Use larger batch on GPU
    bs = BATCH_SIZE * 2 if device.type == "cuda" else BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=bs,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=bs,
                              shuffle=False, num_workers=0)

    # Class weights
    N_total   = len(y_train)
    weights   = np.zeros(NUM_CLASSES, dtype=np.float32)
    unique, counts = np.unique(y_train, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    for c in range(NUM_CLASSES):
        n_c = count_map.get(c, 0)
        weights[c] = min(100.0,
            N_total / (NUM_CLASSES * n_c) if n_c > 0 else 100.0
        )
    class_weights = torch.FloatTensor(weights).to(device)

    # Input dim from X_train shape
    input_dim = X_train.shape[2]

    model = AblationTransformer(
        input_dim   = input_dim,
        d_model     = D_MODEL,
        n_heads     = n_heads,
        n_layers    = n_layers,
        d_ff        = D_FF,
        dropout     = DROPOUT,
        fc_hidden   = FC_HIDDEN,
        n_classes   = NUM_CLASSES,
        use_pos_enc = use_pos_enc,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(
        optimizer, WARMUP_EPOCHS, EPOCHS, LEARNING_RATE
    )

    best_val_f1  = -1.0
    patience_ctr = 0
    best_state   = None
    start_time   = time.time()

    for epoch in range(EPOCHS):
        scheduler.step(epoch)

        # Train
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss   = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                preds   = model(batch_X).argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(batch_y.tolist())

        val_f1 = f1_score(val_labels, val_preds,
                          average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_ctr = 0
            best_state   = {k: v.clone()
                            for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if patience_ctr >= EARLY_STOP_PAT:
            break

    elapsed = time.time() - start_time

    # Test evaluation with best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds   = model(batch_X).argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(batch_y.tolist())

    test_macro_f1 = f1_score(
        test_labels, test_preds,
        average="macro", zero_division=0
    )
    test_acc = sum(p == l for p, l in
                   zip(test_preds, test_labels)) / len(test_labels)

    return {
        "run_name"       : run_name,
        "n_layers"       : n_layers,
        "n_heads"        : n_heads,
        "use_pos_enc"    : use_pos_enc,
        "n_params"       : model.count_parameters(),
        "best_val_f1"    : round(best_val_f1,    4),
        "test_macro_f1"  : round(test_macro_f1,  4),
        "test_accuracy"  : round(test_acc,        4),
        "train_time_s"   : round(elapsed,         1),
        "input_dim"      : input_dim,
    }

# =============================================================
#  6. ABLATION PLOTS
# =============================================================

def plot_ablation_results(all_results: dict,
                           save: bool = True):
    """
    Four-panel figure showing ablation results.
    One panel per ablation dimension.
    Maps to Figure 7 in the paper.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    color = "#1F4E79"

    # ── Panel 1: Window size ──────────────────────────────────
    ws_results = all_results.get("window_size", [])
    if ws_results:
        ws_vals = [r["window_size"] for r in ws_results]
        ws_f1   = [r["test_macro_f1"] for r in ws_results]
        axes[0].bar([str(w) for w in ws_vals], ws_f1,
                    color=color, edgecolor="white", width=0.5)
        axes[0].set_title("7.1 Effect of window size W",
                           fontsize=11)
        axes[0].set_xlabel("Window size W")
        axes[0].set_ylabel("Test macro-F1")
        axes[0].set_ylim(0, 1)
        for i, (w, f) in enumerate(zip(ws_vals, ws_f1)):
            axes[0].text(i, f + 0.02, f"{f:.4f}",
                         ha="center", fontsize=9)
        # Mark default
        default_idx = ws_vals.index(WINDOW_SIZE) \
            if WINDOW_SIZE in ws_vals else 0
        axes[0].get_children()[default_idx].set_color("#E24B4A")

    # ── Panel 2: Encoder layers ───────────────────────────────
    nl_results = all_results.get("n_layers", [])
    if nl_results:
        nl_vals = [r["n_layers"] for r in nl_results]
        nl_f1   = [r["test_macro_f1"] for r in nl_results]
        axes[1].bar([str(n) for n in nl_vals], nl_f1,
                    color=color, edgecolor="white", width=0.5)
        axes[1].set_title("7.2 Effect of encoder layers N",
                           fontsize=11)
        axes[1].set_xlabel("Number of encoder layers N")
        axes[1].set_ylabel("Test macro-F1")
        axes[1].set_ylim(0, 1)
        for i, (n, f) in enumerate(zip(nl_vals, nl_f1)):
            axes[1].text(i, f + 0.02, f"{f:.4f}",
                         ha="center", fontsize=9)
        default_idx = nl_vals.index(N_LAYERS) \
            if N_LAYERS in nl_vals else 0
        axes[1].get_children()[default_idx].set_color("#E24B4A")

    # ── Panel 3: Attention heads ──────────────────────────────
    nh_results = all_results.get("n_heads", [])
    if nh_results:
        nh_vals = [r["n_heads"] for r in nh_results]
        nh_f1   = [r["test_macro_f1"] for r in nh_results]
        axes[2].bar([str(h) for h in nh_vals], nh_f1,
                    color=color, edgecolor="white", width=0.5)
        axes[2].set_title("7.3 Effect of attention heads h",
                           fontsize=11)
        axes[2].set_xlabel("Number of attention heads h")
        axes[2].set_ylabel("Test macro-F1")
        axes[2].set_ylim(0, 1)
        for i, (h, f) in enumerate(zip(nh_vals, nh_f1)):
            axes[2].text(i, f + 0.02, f"{f:.4f}",
                         ha="center", fontsize=9)
        default_idx = nh_vals.index(N_HEADS) \
            if N_HEADS in nh_vals else 0
        axes[2].get_children()[default_idx].set_color("#E24B4A")

    # ── Panel 4: Positional encoding ─────────────────────────
    pe_results = all_results.get("pos_enc", [])
    if pe_results:
        pe_labels = ["With PE\n(sinusoidal)",
                     "Without PE\n(ablated)"]
        pe_f1     = [r["test_macro_f1"] for r in pe_results]
        bars = axes[3].bar(pe_labels, pe_f1,
                           color=[color, "#B5D4F4"],
                           edgecolor="white", width=0.4)
        axes[3].set_title("7.4 Effect of positional encoding",
                           fontsize=11)
        axes[3].set_ylabel("Test macro-F1")
        axes[3].set_ylim(0, 1)
        for i, f in enumerate(pe_f1):
            axes[3].text(i, f + 0.02, f"{f:.4f}",
                         ha="center", fontsize=9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle(
        "Ablation Study — KubernetesTransformer\n"
        "Red bar = default configuration",
        fontsize=13, y=1.02
    )
    plt.tight_layout()

    if save:
        out = os.path.join(ABLATION_DIR, "ablation_results.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot]  Ablation figure → {out}")
    plt.show()
    plt.close()

# =============================================================
#  7. MAIN ABLATION PIPELINE
# =============================================================

def run_stage9() -> dict:
    """
    Execute complete ablation study.

    Runs four sets of experiments varying one parameter at a
    time. Each experiment is a full training run with early
    stopping. Results saved incrementally so partial results
    are preserved if the run is interrupted.

    Returns
    -------
    all_results : dict with results for all four ablations
    """
    print("\n" + "=" * 62)
    print("  STAGE 9 — ABLATION STUDY")
    print("=" * 62 + "\n")

    device = get_device()

    # Load default splits (W=16)
    print(f"[load]  Loading default splits (W={WINDOW_SIZE}) ...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"        Train: {X_train.shape}  "
          f"Val: {X_val.shape}  Test: {X_test.shape}")

    all_results = {}

    # ── Total run estimate ────────────────────────────────────
    total_runs = (len(ABLATION_WINDOWS) +
                  len(ABLATION_LAYERS)  +
                  len(ABLATION_HEADS)   +
                  len(ABLATION_POS_ENC))
    print(f"\n[plan]  Total ablation runs : {total_runs}")
    print(f"        Window sizes        : {ABLATION_WINDOWS}")
    print(f"        Encoder layers      : {ABLATION_LAYERS}")
    print(f"        Attention heads     : {ABLATION_HEADS}")
    print(f"        Positional encoding : {ABLATION_POS_ENC}")
    print(f"\n        Recommended: run on GPU for speed")
    print(f"        Each run uses early stopping "
          f"(patience={EARLY_STOP_PAT})\n")

    # ===========================================================
    # ABLATION 7.1 — WINDOW SIZE
    # ===========================================================
    print(f"\n{'='*62}")
    print(f"  ABLATION 7.1 — WINDOW SIZE  W ∈ {ABLATION_WINDOWS}")
    print(f"{'='*62}")

    ws_results = []
    for W in ABLATION_WINDOWS:
        print(f"\n  Running W={W} ...")

        if W == WINDOW_SIZE:
            # Use already-loaded default splits
            Xtr, ytr = X_train, y_train
            Xvl, yvl = X_val,   y_val
            Xts, yts = X_test,  y_test
        else:
            # Rebuild sequences with different window size
            print(f"  Rebuilding sequences for W={W} ...")
            Xtr, ytr, Xvl, yvl, Xts, yts = \
                rebuild_splits_for_window(W)
            print(f"  Train: {Xtr.shape}  "
                  f"Val: {Xvl.shape}  Test: {Xts.shape}")

        result = run_single_ablation(
            Xtr, ytr, Xvl, yvl, Xts, yts,
            device       = device,
            n_layers     = N_LAYERS,
            n_heads      = N_HEADS,
            use_pos_enc  = True,
            run_name     = f"W={W}",
        )
        result["window_size"] = W
        ws_results.append(result)

        print(f"  W={W:2d}  test macro-F1={result['test_macro_f1']:.4f}"
              f"  val macro-F1={result['best_val_f1']:.4f}"
              f"  time={result['train_time_s']:.0f}s")

    all_results["window_size"] = ws_results

    # Save incrementally
    _save_ablation(all_results)

    # ===========================================================
    # ABLATION 7.2 — ENCODER LAYERS
    # ===========================================================
    print(f"\n{'='*62}")
    print(f"  ABLATION 7.2 — ENCODER LAYERS  N ∈ {ABLATION_LAYERS}")
    print(f"{'='*62}")

    nl_results = []
    for N in ABLATION_LAYERS:
        print(f"\n  Running N={N} ...")
        result = run_single_ablation(
            X_train, y_train, X_val, y_val, X_test, y_test,
            device       = device,
            n_layers     = N,
            n_heads      = N_HEADS,
            use_pos_enc  = True,
            run_name     = f"N={N}",
        )
        result["window_size"] = WINDOW_SIZE
        nl_results.append(result)

        print(f"  N={N}  test macro-F1={result['test_macro_f1']:.4f}"
              f"  params={result['n_params']:,}"
              f"  time={result['train_time_s']:.0f}s")

    all_results["n_layers"] = nl_results
    _save_ablation(all_results)

    # ===========================================================
    # ABLATION 7.3 — ATTENTION HEADS
    # ===========================================================
    print(f"\n{'='*62}")
    print(f"  ABLATION 7.3 — ATTENTION HEADS  h ∈ {ABLATION_HEADS}")
    print(f"{'='*62}")

    nh_results = []
    for H in ABLATION_HEADS:
        print(f"\n  Running h={H} ...")
        result = run_single_ablation(
            X_train, y_train, X_val, y_val, X_test, y_test,
            device       = device,
            n_layers     = N_LAYERS,
            n_heads      = H,
            use_pos_enc  = True,
            run_name     = f"h={H}",
        )
        result["window_size"] = WINDOW_SIZE
        nh_results.append(result)

        print(f"  h={H}  test macro-F1={result['test_macro_f1']:.4f}"
              f"  time={result['train_time_s']:.0f}s")

    all_results["n_heads"] = nh_results
    _save_ablation(all_results)

    # ===========================================================
    # ABLATION 7.4 — POSITIONAL ENCODING
    # ===========================================================
    print(f"\n{'='*62}")
    print(f"  ABLATION 7.4 — POSITIONAL ENCODING")
    print(f"{'='*62}")

    pe_results = []
    for use_pe in ABLATION_POS_ENC:
        label = "with_PE" if use_pe else "without_PE"
        print(f"\n  Running {label} ...")
        result = run_single_ablation(
            X_train, y_train, X_val, y_val, X_test, y_test,
            device       = device,
            n_layers     = N_LAYERS,
            n_heads      = N_HEADS,
            use_pos_enc  = use_pe,
            run_name     = label,
        )
        result["window_size"] = WINDOW_SIZE
        pe_results.append(result)

        print(f"  PE={use_pe}  "
              f"test macro-F1={result['test_macro_f1']:.4f}"
              f"  time={result['train_time_s']:.0f}s")

    all_results["pos_enc"] = pe_results
    _save_ablation(all_results)

    # ===========================================================
    # SUMMARY TABLE
    # ===========================================================
    _print_ablation_summary(all_results)

    # ===========================================================
    # PLOT
    # ===========================================================
    plot_ablation_results(all_results, save=True)

    print(f"\n[stage9] ✓ Ablation study complete.")
    print(f"         Results → {ABLATION_DIR}/ablation_results.json")
    print(f"         Figure  → {ABLATION_DIR}/ablation_results.png\n")

    return all_results


def _save_ablation(results: dict):
    """Save ablation results incrementally as JSON."""
    out = os.path.join(ABLATION_DIR, "ablation_results.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    safe = {}
    for key, runs in results.items():
        safe[key] = []
        for run in runs:
            safe[key].append(
                {k: convert(v) for k, v in run.items()}
            )

    with open(out, "w") as f:
        json.dump(safe, f, indent=2)


def _print_ablation_summary(all_results: dict):
    """Print summary table of all ablation results."""
    sep = "=" * 62
    default_f1 = None

    # Find default config result (W=16, N=4, h=8, PE=True)
    for r in all_results.get("window_size", []):
        if r.get("window_size") == WINDOW_SIZE:
            default_f1 = r["test_macro_f1"]
            break

    print(f"\n{sep}")
    print("  ABLATION STUDY SUMMARY")
    print(sep)

    sections = [
        ("7.1 Window size",
         "window_size", "window_size", ABLATION_WINDOWS),
        ("7.2 Encoder layers",
         "n_layers",    "n_layers",    ABLATION_LAYERS),
        ("7.3 Attention heads",
         "n_heads",     "n_heads",     ABLATION_HEADS),
    ]

    for title, key, field, values in sections:
        print(f"\n  {title}")
        print(f"  {'Config':<12} {'Test Macro-F1':>14} "
              f"{'Params':>10}  Note")
        print(f"  {'-'*12} {'-'*14} {'-'*10}  {'-'*10}")
        for r in all_results.get(key, []):
            val     = r.get(field, "?")
            f1      = r["test_macro_f1"]
            params  = r.get("n_params", 0)
            is_def  = (val == {
                "window_size": WINDOW_SIZE,
                "n_layers"   : N_LAYERS,
                "n_heads"    : N_HEADS,
            }.get(field, val))
            note = "← default" if is_def else ""
            print(f"  {str(val):<12} {f1:>14.4f} "
                  f"{params:>10,}  {note}")

    # Positional encoding
    print(f"\n  7.4 Positional encoding")
    print(f"  {'Config':<14} {'Test Macro-F1':>14}  Note")
    print(f"  {'-'*14} {'-'*14}  {'-'*10}")
    for r in all_results.get("pos_enc", []):
        label  = "With PE" if r["use_pos_enc"] else "Without PE"
        f1     = r["test_macro_f1"]
        note   = "← default" if r["use_pos_enc"] else ""
        print(f"  {label:<14} {f1:>14.4f}  {note}")

    print(f"\n{sep}\n")


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  Mac (slow but works):
#    python scripts/stage9_ablation.py
#
#  BITS Pilani GPU cluster (recommended):
#    1. Copy scripts/ and data/ to cluster
#    2. Change ENVIRONMENT = "cluster" in config.py
#    3. Submit as SLURM job:
#
#       #!/bin/bash
#       #SBATCH --job-name=k8s_ablation
#       #SBATCH --partition=gpu
#       #SBATCH --gres=gpu:1
#       #SBATCH --cpus-per-task=4
#       #SBATCH --mem=16G
#       #SBATCH --time=06:00:00
#       #SBATCH --output=logs/ablation_%j.out
#
#       module load python/3.10
#       module load cuda/11.8
#       source ~/kubernetes_anomaly_detection/venv/bin/activate
#       python scripts/stage9_ablation.py
#
#    Submit: sbatch job_ablation.sh
#    Status: squeue -u f20221516
#
#  Results are saved incrementally — if interrupted,
#  partial results are preserved in ablation_results.json.
#
# =============================================================

if __name__ == "__main__":
    run_stage9()