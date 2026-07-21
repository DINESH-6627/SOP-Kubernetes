# =============================================================
#  stage6_train.py
#  Stage 6 — Model Training
#
#  What this script does:
#    1. Builds model and moves to device (MPS/CUDA/CPU)
#    2. Sets up AdamW optimizer + cosine annealing with warmup
#    3. Runs training loop with weighted cross-entropy loss
#    4. Evaluates on validation set after every epoch
#    5. Early stopping on validation macro-F1 (patience=10)
#    6. Saves best checkpoint when val macro-F1 improves
#    7. Logs training curves (loss + macro-F1 per epoch)
#    8. Plots and saves training curves at end of training
#
#  Training protocol (paper Section 4.6):
#    Optimizer  : AdamW (lr=3e-4, weight_decay=1e-4)
#    Scheduler  : Linear warmup (5 epochs) + cosine annealing
#    Loss       : Weighted cross-entropy (class imbalance)
#    Early stop : patience=10 on val macro-F1
#    Primary metric : macro-F1 (treats all 11 classes equally)
#
#  Input  : DataLoaders + class_weights from stage4
#           Model from stage5
#  Output : Best model checkpoint saved to checkpoints/
#           Training curves saved to results/
# =============================================================

import os
import sys
import math
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
import json
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# [KAGGLE]  PROJECT_ROOT = "/kaggle/working"
# [CLUSTER] PROJECT_ROOT = "/home/<username>/kubernetes_anomaly_detection"

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.py")
spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load config module from: {CONFIG_PATH}")
project_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_config)

CHECKPOINT_DIR = project_config.CHECKPOINT_DIR
RESULTS_DIR = project_config.RESULTS_DIR
LOG_DIR = project_config.LOG_DIR
EPOCHS = project_config.EPOCHS
LEARNING_RATE = project_config.LEARNING_RATE
WEIGHT_DECAY = project_config.WEIGHT_DECAY
WARMUP_EPOCHS = project_config.WARMUP_EPOCHS
EARLY_STOP_PAT = project_config.EARLY_STOP_PAT
BATCH_SIZE = project_config.BATCH_SIZE
NUM_CLASSES = project_config.NUM_CLASSES
RANDOM_SEED = project_config.RANDOM_SEED
D_MODEL = project_config.D_MODEL
N_HEADS = project_config.N_HEADS
N_LAYERS = project_config.N_LAYERS
D_FF = project_config.D_FF
DROPOUT = project_config.DROPOUT
FC_HIDDEN = project_config.FC_HIDDEN
WINDOW_SIZE = project_config.WINDOW_SIZE

INPUT_DIM = 39   # confirmed after Stage 2

# =============================================================
#  1. LEARNING RATE SCHEDULER WITH WARMUP
# =============================================================

class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine annealing decay.

    During warmup (first warmup_epochs):
      lr = base_lr * (epoch / warmup_epochs)

    After warmup:
      lr = base_lr * 0.5 * (1 + cos(pi * progress))
      where progress = (epoch - warmup) / (total - warmup)

    This prevents the large gradient updates that can
    destabilize transformer training in early epochs when
    weights are randomly initialized.
    """

    def __init__(self,
                 optimizer:     optim.Optimizer,
                 warmup_epochs: int,
                 total_epochs:  int,
                 base_lr:       float,
                 min_lr:        float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr

    def step(self, epoch: int):
        """Update learning rate for given epoch (0-indexed)."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = ((epoch - self.warmup_epochs) /
                        max(1, self.total_epochs -
                            self.warmup_epochs))
            lr = self.min_lr + 0.5 * (
                self.base_lr - self.min_lr
            ) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

# =============================================================
#  2. TRAINING EPOCH
# =============================================================

def train_one_epoch(model:        nn.Module,
                    loader:       torch.utils.data.DataLoader,
                    optimizer:    optim.Optimizer,
                    criterion:    nn.Module,
                    device:       torch.device) -> tuple:
    """
    Run one full training epoch.

    Returns
    -------
    avg_loss  : float  mean loss over all batches
    macro_f1  : float  macro-averaged F1 over all batches
    """
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)             # (B, n_classes)
        loss   = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(
        all_labels, all_preds,
        average="macro",
        zero_division=0
    )
    return avg_loss, macro_f1

# =============================================================
#  3. VALIDATION EPOCH
# =============================================================

def evaluate(model:    nn.Module,
             loader:   torch.utils.data.DataLoader,
             criterion: nn.Module,
             device:   torch.device) -> tuple:
    """
    Evaluate model on val or test loader.

    Returns
    -------
    avg_loss    : float
    macro_f1    : float
    all_preds   : list  — predicted labels
    all_labels  : list  — true labels
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss   = criterion(logits, batch_y)

            total_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(
        all_labels, all_preds,
        average="macro",
        zero_division=0
    )
    return avg_loss, macro_f1, all_preds, all_labels

# =============================================================
#  4. TRAINING CURVES PLOT
# =============================================================

def plot_training_curves(history: dict,
                         save: bool = True):
    """
    Plot loss and macro-F1 curves for train and validation.
    Saves to results/training_curves.png
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"],
                 label="Train loss", color="#378ADD",
                 linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"],
                 label="Val loss", color="#E24B4A",
                 linewidth=1.5)
    if history.get("best_epoch"):
        axes[0].axvline(x=history["best_epoch"],
                        color="gray", linestyle="--",
                        linewidth=1, label="Best epoch")
    axes[0].set_title("Training and validation loss",
                      fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted cross-entropy loss")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(linestyle="--", alpha=0.3)

    # Macro-F1 plot
    axes[1].plot(epochs, history["train_f1"],
                 label="Train macro-F1", color="#378ADD",
                 linewidth=1.5)
    axes[1].plot(epochs, history["val_f1"],
                 label="Val macro-F1", color="#E24B4A",
                 linewidth=1.5)
    if history.get("best_epoch"):
        axes[1].axvline(x=history["best_epoch"],
                        color="gray", linestyle="--",
                        linewidth=1, label="Best epoch")
    axes[1].set_title("Training and validation macro-F1",
                      fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-averaged F1 score")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(linestyle="--", alpha=0.3)

    plt.suptitle(
        f"Training curves — KubernetesTransformer\n"
        f"d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, "
        f"W={WINDOW_SIZE}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "training_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot]  Training curves saved → {out}")
    try:
        plt.show()
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        plt.close()

# =============================================================
#  5. MAIN TRAINING LOOP
# =============================================================

def run_stage6(train_loader,
               val_loader,
               class_weights: torch.Tensor,
               device:        torch.device,
               model=None) -> tuple:
    """
    Execute complete Stage 6 training pipeline.

    Parameters
    ----------
    train_loader  : DataLoader
    val_loader    : DataLoader
    class_weights : torch.Tensor  (NUM_CLASSES,) on device
    device        : torch.device
    model         : KubernetesTransformer (built externally)
                    If None, builds from config

    Returns
    -------
    model    : best trained model (checkpoint loaded)
    history  : dict with full training log
    """
    print("\n" + "=" * 62)
    print("  STAGE 6 — MODEL TRAINING")
    print("=" * 62 + "\n")

    # ── Build model if not provided ───────────────────────────
    if model is None:
        from scripts.stage5_model import build_model
        model, _ = build_model(device)
    else:
        model = model.to(device)

    # ── Loss function ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )

    # ── LR scheduler ─────────────────────────────────────────
    scheduler = WarmupCosineScheduler(
        optimizer     = optimizer,
        warmup_epochs = WARMUP_EPOCHS,
        total_epochs  = EPOCHS,
        base_lr       = LEARNING_RATE,
    )

    print(f"  Optimizer    : AdamW  "
          f"(lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler    : Linear warmup ({WARMUP_EPOCHS} ep)"
          f" + cosine annealing")
    print(f"  Loss         : Weighted cross-entropy")
    print(f"  Early stop   : patience={EARLY_STOP_PAT} "
          f"on val macro-F1")
    print(f"  Max epochs   : {EPOCHS}")
    print(f"  Device       : {device}")
    print(f"  Parameters   : {model.count_parameters():,}\n")

    # ── Training state ────────────────────────────────────────
    history = {
        "train_loss" : [],
        "train_f1"   : [],
        "val_loss"   : [],
        "val_f1"     : [],
        "lr"         : [],
        "best_epoch" : None,
    }

    best_val_f1   = -1.0
    patience_ctr  = 0
    best_ckpt     = os.path.join(
        CHECKPOINT_DIR, "best_model.pt"
    )

    print(f"  {'Epoch':>6}  {'LR':>10}  "
          f"{'Train Loss':>11}  {'Train F1':>9}  "
          f"{'Val Loss':>9}  {'Val F1':>8}  {'':>6}")
    print(f"  {'-'*6}  {'-'*10}  "
          f"{'-'*11}  {'-'*9}  "
          f"{'-'*9}  {'-'*8}  {'-'*6}")

    # ── Main loop ─────────────────────────────────────────────
    for epoch in range(EPOCHS):

        # Update learning rate
        current_lr = scheduler.step(epoch)

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Log
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        # Best model checkpoint
        improved = val_f1 > best_val_f1
        marker   = " ← best" if improved else ""

        if improved:
            best_val_f1 = val_f1
            patience_ctr = 0
            history["best_epoch"] = epoch + 1
            torch.save({
                "epoch"       : epoch + 1,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_f1"      : val_f1,
                "val_loss"    : val_loss,
                "config"      : {
                    "input_dim" : INPUT_DIM,
                    "d_model"   : D_MODEL,
                    "n_heads"   : N_HEADS,
                    "n_layers"  : N_LAYERS,
                    "d_ff"      : D_FF,
                    "dropout"   : DROPOUT,
                    "fc_hidden" : FC_HIDDEN,
                    "n_classes" : NUM_CLASSES,
                    "window_size": WINDOW_SIZE,
                }
            }, best_ckpt)
        else:
            patience_ctr += 1

        # Print epoch row
        print(f"  {epoch+1:>6}  {current_lr:>10.2e}  "
              f"{train_loss:>11.4f}  {train_f1:>9.4f}  "
              f"{val_loss:>9.4f}  {val_f1:>8.4f}  {marker}")

        # Early stopping
        if patience_ctr >= EARLY_STOP_PAT:
            print(f"\n[train]  Early stopping triggered.")
            print(f"         No val macro-F1 improvement for "
                  f"{EARLY_STOP_PAT} consecutive epochs.")
            break

    # ── Load best checkpoint ──────────────────────────────────
    print(f"\n[train]  Loading best checkpoint ...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"         Best epoch    : {ckpt['epoch']}")
    print(f"         Best val F1   : {ckpt['val_f1']:.4f}")
    print(f"         Checkpoint    : {best_ckpt}")

    # ── Save training history ─────────────────────────────────
    hist_path = os.path.join(LOG_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[save]   Training history → {hist_path}")

    # ── Plot curves ───────────────────────────────────────────
    plot_training_curves(history, save=True)

    print(f"\n[stage6] ✓ Training complete.")
    print(f"         Best val macro-F1 : {best_val_f1:.4f}")
    print(f"         Best epoch        : "
          f"{history['best_epoch']}")
    print(f"         → Pass model to "
          f"stage8_evaluation.run_stage8()\n")

    return model, history


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  Terminal (full pipeline from saved arrays):
#    python scripts/stage6_train.py
#
#  From notebook after stages 4 and 5:
#    from scripts.stage6_train import run_stage6
#    model, history = run_stage6(
#        train_loader, val_loader,
#        class_weights, device, model
#    )
#
# =============================================================

if __name__ == "__main__":
    # Load everything from saved files
    from scripts.stage4_dataloader import run_stage4
    from scripts.stage5_model import build_model

    train_loader, val_loader, test_loader, \
        class_weights, device, datasets = run_stage4()

    model, device = build_model(device)

    model, history = run_stage6(
        train_loader  = train_loader,
        val_loader    = val_loader,
        class_weights = class_weights,
        device        = device,
        model         = model,
    )