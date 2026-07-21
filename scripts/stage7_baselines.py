# =============================================================
#  stage7_baselines.py
#  Stage 7 — Baseline Model Training and Evaluation
#
#  Baselines selected based on novelty requirements:
#
#  1. Random Forest  — best classical per-flow baseline.
#     Operates on INDIVIDUAL flows (no sequences).
#     Directly demonstrates the per-flow independence
#     limitation that Novelty 1 addresses.
#     Used in original Sever & Dogan (2023) paper.
#
#  2. MLP (3-layer)  — deep learning per-flow baseline.
#     Operates on INDIVIDUAL flows (no sequences).
#     Bridges classical ML and sequence-aware deep learning.
#
#  3. LSTM           — recurrent sequence baseline.
#     Operates on SEQUENCES — same (N, W, D) input as transformer.
#     Critical for Novelty 1: if transformer > LSTM, this proves
#     the attention mechanism specifically drives improvement,
#     not just sequence modeling in general.
#
#  SVM skipped — slow on Mac CPU, results citable from
#  Aly et al. [32] who ran it on this exact dataset.
#
#  All baselines evaluated on the SAME test split as the
#  transformer for fair comparison (Novelty 3).
#
#  Input  : numpy arrays from stage3 (X_train, y_train, etc.)
#  Output : baseline_results.pkl saved to results/
# =============================================================

import os
import sys
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)
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

DATA_DIR = project_config.DATA_DIR
RESULTS_DIR = project_config.RESULTS_DIR
CHECKPOINT_DIR = project_config.CHECKPOINT_DIR
LOG_DIR = project_config.LOG_DIR
RF_N_ESTIMATORS = project_config.RF_N_ESTIMATORS
RF_MAX_DEPTH = project_config.RF_MAX_DEPTH
MLP_HIDDEN_SIZES = project_config.MLP_HIDDEN_SIZES
MLP_MAX_ITER = project_config.MLP_MAX_ITER
LSTM_HIDDEN = project_config.LSTM_HIDDEN
LSTM_LAYERS = project_config.LSTM_LAYERS
LSTM_DROPOUT = project_config.LSTM_DROPOUT
BATCH_SIZE = project_config.BATCH_SIZE
NUM_CLASSES = project_config.NUM_CLASSES
RANDOM_SEED = project_config.RANDOM_SEED
CLASS_NAMES = project_config.CLASS_NAMES
LEARNING_RATE = project_config.LEARNING_RATE
EPOCHS = project_config.EPOCHS
EARLY_STOP_PAT = project_config.EARLY_STOP_PAT
WINDOW_SIZE = project_config.WINDOW_SIZE

INPUT_DIM = 39

# =============================================================
#  HELPER — LOAD SPLITS
# =============================================================

def load_splits() -> tuple:
    """Load all numpy arrays saved by stage3."""
    print(f"[load]  Loading splits from {DATA_DIR} ...")
    arrays = {}
    for name in ["X_train", "y_train", "X_val",
                 "y_val", "X_test", "y_test"]:
        path = os.path.join(DATA_DIR, f"{name}.npy")
        arrays[name] = np.load(path)
        print(f"        {name:<10} → {arrays[name].shape}")
    return (arrays["X_train"], arrays["y_train"],
            arrays["X_val"],   arrays["y_val"],
            arrays["X_test"],  arrays["y_test"])


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flatten (N, W, D) sequences to (N, W*D) for per-flow
    baselines that do not model sequences.

    For RF and MLP we use the MEAN across the W dimension
    instead of full flatten — this gives a compact per-sequence
    feature vector that is more informative than random flattening
    and avoids the curse of dimensionality with W*D=624 features.

    Mean pooling across W: (N, W, D) → (N, D)
    This is still "per-flow equivalent" since it loses temporal
    ordering — exactly the limitation we are demonstrating.
    """
    return X.mean(axis=1)   # (N, D) — temporal info discarded


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    model_name: str) -> dict:
    """Compute and print all evaluation metrics."""
    macro_f1    = f1_score(y_true, y_pred,
                           average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred,
                           average="weighted", zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)

    print(f"\n  {model_name} — Test Results:")
    print(f"  {'Macro F1':<20} : {macro_f1:.4f}")
    print(f"  {'Weighted F1':<20} : {weighted_f1:.4f}")
    print(f"  {'Accuracy':<20} : {accuracy:.4f}")

    # Per-class report
    print(f"\n  Per-class report:")
    target_names = [CLASS_NAMES.get(i, str(i))
                    for i in range(NUM_CLASSES)]
    print(classification_report(
        y_true, y_pred,
        target_names = target_names,
        zero_division = 0,
        digits = 4
    ))

    return {
        "model"       : model_name,
        "macro_f1"    : round(macro_f1, 4),
        "weighted_f1" : round(weighted_f1, 4),
        "accuracy"    : round(accuracy, 4),
        "y_pred"      : y_pred.tolist(),
        "y_true"      : y_true.tolist(),
    }

# =============================================================
#  BASELINE 1 — RANDOM FOREST
# =============================================================

def train_random_forest(X_train_flat: np.ndarray,
                        y_train:      np.ndarray,
                        X_test_flat:  np.ndarray,
                        y_test:       np.ndarray) -> dict:
    """
    Random Forest classifier operating on mean-pooled
    per-sequence feature vectors.

    Per-flow baseline: temporal ordering is discarded via
    mean pooling — this directly demonstrates the limitation
    of stateless per-flow classifiers (paper Section 2.2).

    Config: 100 trees, no max depth, class_weight balanced.
    """
    print(f"\n{'='*62}")
    print(f"  BASELINE 1 — RANDOM FOREST")
    print(f"{'='*62}")
    print(f"  Trees          : {RF_N_ESTIMATORS}")
    print(f"  Max depth      : {RF_MAX_DEPTH or 'unlimited'}")
    print(f"  Class weight   : balanced")
    print(f"  Input shape    : {X_train_flat.shape}")
    print(f"  Sequence-aware : NO (mean-pooled features)")

    rf = RandomForestClassifier(
        n_estimators = RF_N_ESTIMATORS,
        max_depth    = RF_MAX_DEPTH,
        class_weight = "balanced",
        random_state = RANDOM_SEED,
        n_jobs       = -1,      # use all CPU cores
        verbose      = 0,
    )

    print(f"\n[RF]  Training ...")
    rf.fit(X_train_flat, y_train)
    print(f"[RF]  Training complete.")

    y_pred = rf.predict(X_test_flat)
    results = compute_metrics(y_test, y_pred, "Random Forest")

    # Save model
    rf_path = os.path.join(CHECKPOINT_DIR, "rf_model.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"[RF]  Model saved → {rf_path}")

    return results

# =============================================================
#  BASELINE 2 — MLP
# =============================================================

def train_mlp(X_train_flat: np.ndarray,
              y_train:      np.ndarray,
              X_test_flat:  np.ndarray,
              y_test:       np.ndarray) -> dict:
    """
    Multi-Layer Perceptron operating on mean-pooled
    per-sequence feature vectors.

    Per-flow deep learning baseline: no temporal modeling.
    Architecture: 256 → 128 → 64 → 11 classes.
    """
    print(f"\n{'='*62}")
    print(f"  BASELINE 2 — MLP (3-layer)")
    print(f"{'='*62}")
    print(f"  Hidden sizes   : {MLP_HIDDEN_SIZES}")
    print(f"  Max iterations : {MLP_MAX_ITER}")
    print(f"  Input shape    : {X_train_flat.shape}")
    print(f"  Sequence-aware : NO (mean-pooled features)")

    mlp = MLPClassifier(
        hidden_layer_sizes = MLP_HIDDEN_SIZES,
        activation         = "relu",
        solver             = "adam",
        alpha              = 1e-4,
        batch_size         = 64,
        learning_rate_init = 1e-3,
        max_iter           = MLP_MAX_ITER,
        random_state       = RANDOM_SEED,
        early_stopping     = True,
        validation_fraction= 0.15,
        n_iter_no_change   = 10,
        verbose            = False,
    )

    print(f"\n[MLP] Training ...")
    mlp.fit(X_train_flat, y_train)
    print(f"[MLP] Training complete. "
          f"Iterations: {mlp.n_iter_}")

    y_pred  = mlp.predict(X_test_flat)
    results = compute_metrics(y_test, y_pred, "MLP")

    mlp_path = os.path.join(CHECKPOINT_DIR, "mlp_model.pkl")
    with open(mlp_path, "wb") as f:
        pickle.dump(mlp, f)
    print(f"[MLP] Model saved → {mlp_path}")

    return results

# =============================================================
#  BASELINE 3 — LSTM
# =============================================================

class LSTMClassifier(nn.Module):
    """
    LSTM-based sequence classifier.

    Takes the same (B, W, D) input as the transformer.
    Uses LSTM hidden state from the last time step for
    classification — this is the key architectural difference:
    LSTM compresses all sequence history into a fixed hidden
    state vector, while the transformer attends to all
    positions simultaneously.

    Architecture:
      Input  : (B, W, D) = (B, 16, 39)
      LSTM   : D → hidden_size, n_layers, bidirectional=False
      Output : last hidden state → FC → 11 classes
    """

    def __init__(self,
                 input_dim:   int = INPUT_DIM,
                 hidden_size: int = LSTM_HIDDEN,
                 n_layers:    int = LSTM_LAYERS,
                 dropout:     float = LSTM_DROPOUT,
                 n_classes:   int = NUM_CLASSES):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            dropout      = dropout if n_layers > 1 else 0.0,
            batch_first  = True,
            bidirectional= False,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, W, D)
        Returns logits : (B, n_classes)
        """
        # lstm_out : (B, W, hidden_size)
        # hidden   : (n_layers, B, hidden_size)
        lstm_out, (hidden, _) = self.lstm(x)

        # Use last layer hidden state
        last_hidden = hidden[-1]              # (B, hidden_size)
        last_hidden = self.dropout(last_hidden)
        logits      = self.classifier(last_hidden)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


def train_lstm(X_train: np.ndarray,
               y_train: np.ndarray,
               X_val:   np.ndarray,
               y_val:   np.ndarray,
               X_test:  np.ndarray,
               y_test:  np.ndarray,
               device:  torch.device) -> dict:
    """
    Train LSTM baseline on the same sequences as the transformer.
    Uses same training protocol — AdamW, weighted CE, early stop.
    This ensures any performance difference is due to architecture
    not training procedure.
    """
    print(f"\n{'='*62}")
    print(f"  BASELINE 3 — LSTM")
    print(f"{'='*62}")
    print(f"  Hidden size    : {LSTM_HIDDEN}")
    print(f"  Layers         : {LSTM_LAYERS}")
    print(f"  Dropout        : {LSTM_DROPOUT}")
    print(f"  Input shape    : {X_train.shape}")
    print(f"  Sequence-aware : YES (recurrent hidden state)")

    # ── Dataset and DataLoader ────────────────────────────────
    from scripts.stage4_dataloader import KubernetesSequenceDataset

    train_ds = KubernetesSequenceDataset(X_train, y_train)
    val_ds   = KubernetesSequenceDataset(X_val,   y_val)
    test_ds  = KubernetesSequenceDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Class weights ─────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────
    model     = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=1e-4)

    print(f"\n[LSTM] Parameters: {model.count_parameters():,}")
    print(f"[LSTM] Training ...")
    print(f"\n  {'Epoch':>6}  {'Train F1':>9}  "
          f"{'Val F1':>8}  {'':>6}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*6}")

    best_val_f1  = -1.0
    patience_ctr = 0
    best_state   = None

    for epoch in range(EPOCHS):
        # Train
        model.train()
        all_preds, all_labels = [], []
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
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.cpu().numpy().tolist())

        train_f1 = f1_score(all_labels, all_preds,
                            average="macro", zero_division=0)

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits  = model(batch_X)
                preds   = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_labels.extend(batch_y.cpu().numpy().tolist())

        val_f1 = f1_score(val_labels, val_preds,
                          average="macro", zero_division=0)

        improved = val_f1 > best_val_f1
        marker   = " ← best" if improved else ""

        if improved:
            best_val_f1  = val_f1
            patience_ctr = 0
            best_state   = {k: v.clone()
                            for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        print(f"  {epoch+1:>6}  {train_f1:>9.4f}  "
              f"{val_f1:>8.4f}  {marker}")

        if patience_ctr >= EARLY_STOP_PAT:
            print(f"\n[LSTM] Early stopping at epoch {epoch+1}")
            break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[LSTM] Best val macro-F1: {best_val_f1:.4f}")

    # Test evaluation
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits  = model(batch_X)
            preds   = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds.tolist())

    results = compute_metrics(
        np.array(y_test), np.array(test_preds), "LSTM"
    )
    results["best_val_f1"] = round(best_val_f1, 4)

    lstm_path = os.path.join(CHECKPOINT_DIR, "lstm_model.pt")
    torch.save(model.state_dict(), lstm_path)
    print(f"[LSTM] Model saved → {lstm_path}")

    return results

# =============================================================
#  COMPARISON SUMMARY TABLE
# =============================================================

def print_comparison_table(results_list: list,
                            transformer_val_f1: float = None):
    """
    Print a summary comparison table of all models.
    This maps directly to Table 4 in the paper.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("  MODEL COMPARISON TABLE  (Test Set)")
    print(sep)
    print(f"  {'Model':<22} {'Macro-F1':>10} "
          f"{'Weighted-F1':>12} {'Accuracy':>10}  "
          f"{'Seq-Aware':>10}")
    print(f"  {'-'*22} {'-'*10} "
          f"{'-'*12} {'-'*10}  {'-'*10}")

    seq_aware = {
        "Random Forest" : "No",
        "MLP"           : "No",
        "LSTM"          : "Yes (recurrent)",
        "Transformer"   : "Yes (attention)",
    }

    for r in results_list:
        name = r["model"]
        print(f"  {name:<22} {r['macro_f1']:>10.4f} "
              f"{r['weighted_f1']:>12.4f} "
              f"{r['accuracy']:>10.4f}  "
              f"{seq_aware.get(name, '—'):>10}")

    print(f"\n  Primary metric: Macro-F1 "
          f"(treats all 11 classes equally)")
    print(f"  Test split    : same for all models")
    print(f"  Dataset       : Sever & Dogan (2023)")
    print(f"{sep}\n")

# =============================================================
#  MAIN PIPELINE
# =============================================================

def run_stage7(device: torch.device = None) -> dict:
    """
    Train and evaluate all three baseline models.

    Returns
    -------
    baseline_results : dict with results for RF, MLP, LSTM
    """
    print("\n" + "=" * 62)
    print("  STAGE 7 — BASELINE MODEL TRAINING")
    print("=" * 62 + "\n")

    # Device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"[device] {device}")

    # Load splits
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test) = load_splits()

    # Flatten for per-flow baselines (mean pool across W)
    X_train_flat = flatten_sequences(X_train)
    X_test_flat  = flatten_sequences(X_test)

    print(f"\n[prep]  Per-flow baseline input: "
          f"{X_train_flat.shape} "
          f"(mean-pooled, temporal order discarded)")
    print(f"[prep]  Sequence baseline input : "
          f"{X_train.shape} (full sequences)")

    all_results = []

    # ── Baseline 1: Random Forest ─────────────────────────────
    rf_results = train_random_forest(
        X_train_flat, y_train,
        X_test_flat,  y_test
    )
    all_results.append(rf_results)

    # ── Baseline 2: MLP ───────────────────────────────────────
    mlp_results = train_mlp(
        X_train_flat, y_train,
        X_test_flat,  y_test
    )
    all_results.append(mlp_results)

    # ── Baseline 3: LSTM ──────────────────────────────────────
    lstm_results = train_lstm(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        device
    )
    all_results.append(lstm_results)

    # ── Comparison table ──────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save results ──────────────────────────────────────────
    results_dict = {r["model"]: r for r in all_results}
    out_path = os.path.join(RESULTS_DIR, "baseline_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"[save]  Baseline results → {out_path}")

    out_json = os.path.join(RESULTS_DIR, "baseline_results.json")
    json_safe = {}
    for model_name, res in results_dict.items():
        json_safe[model_name] = {
            k: v for k, v in res.items()
            if k not in ["y_pred", "y_true"]
        }
    with open(out_json, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"[save]  Baseline summary → {out_json}")

    print(f"\n[stage7] ✓ Complete.")
    print(f"         → Run stage8_evaluation.py to generate "
          f"full comparison including transformer\n")

    return results_dict


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  Terminal:
#    python scripts/stage7_baselines.py
#
#  From notebook:
#    from scripts.stage7_baselines import run_stage7
#    baseline_results = run_stage7(device)
#
# =============================================================

if __name__ == "__main__":
    run_stage7()