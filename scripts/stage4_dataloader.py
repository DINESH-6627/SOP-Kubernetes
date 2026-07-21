# =============================================================
#  stage4_dataloader.py
#  Stage 4 — PyTorch Dataset and DataLoader Construction
#
#  What this script does:
#    1. Defines KubernetesSequenceDataset — PyTorch Dataset class
#       wrapping the (N, W, D) numpy arrays from Stage 3
#    2. Computes class weights for weighted cross-entropy loss
#       inversely proportional to class frequency in train set
#    3. Creates DataLoaders for train / val / test splits
#    4. Runs a sanity check — verifies one batch loads correctly
#    5. Detects and reports the available compute device
#       (mps / cuda / cpu)
#
#  Input  : numpy arrays saved by stage3_sequences.py
#  Output : (train_loader, val_loader, test_loader,
#             class_weights, device)
# =============================================================

import os
import sys
import importlib.util
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
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
BATCH_SIZE = project_config.BATCH_SIZE
NUM_CLASSES = project_config.NUM_CLASSES
RANDOM_SEED = project_config.RANDOM_SEED
CLASS_NAMES = project_config.CLASS_NAMES

# =============================================================
#  1. DEVICE DETECTION
# =============================================================

def get_device() -> torch.device:
    """
    Auto-detect the best available compute device.
      Apple Silicon Mac → mps  (Metal GPU — confirmed working)
      NVIDIA GPU        → cuda
      Fallback          → cpu

    Returns torch.device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[device] Apple Silicon MPS detected → using GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA GPU detected → {gpu_name}")
    else:
        device = torch.device("cpu")
        print(f"[device] No GPU detected → using CPU")
        print(f"         Training will be slower on CPU.")
        print(f"         Consider using college GPU when available.")

    print(f"[device] Active device: {device}")
    return device

# =============================================================
#  2. PYTORCH DATASET
# =============================================================

class KubernetesSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping Kubernetes network flow sequences.

    Each item is a (sequence, label) pair where:
      sequence : torch.FloatTensor  shape (W, D) = (16, 39)
      label    : torch.LongTensor   scalar

    The sequence shape (W, D) maps directly to the transformer
    encoder input: W time steps, each with D features.

    Parameters
    ----------
    X : np.ndarray  shape (N, W, D)
    y : np.ndarray  shape (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert to tensors once at init — faster than per-item
        self.X = torch.FloatTensor(X)   # (N, W, D)
        self.y = torch.LongTensor(y)    # (N,)
        self.n_sequences = len(X)
        self.window_size = X.shape[1]   # W
        self.n_features  = X.shape[2]   # D

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return (f"KubernetesSequenceDataset("
                f"n={self.n_sequences}, "
                f"W={self.window_size}, "
                f"D={self.n_features})")

# =============================================================
#  3. CLASS WEIGHT COMPUTATION
# =============================================================

def compute_class_weights(y_train: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class
    frequency in the training set.

    Formula:
      w_c = N_total / (NUM_CLASSES * N_c)

    where N_total is total training sequences and N_c is the
    count of class c in training. Rare attack classes receive
    proportionally higher weights, forcing the model to pay
    more attention to errors on those classes.

    Weights are capped at MAX_WEIGHT to prevent training
    instability from extremely large weights on rare classes.

    Returns
    -------
    class_weights : torch.Tensor  shape (NUM_CLASSES,)
                    on the target device
    """
    MAX_WEIGHT  = 100.0    # cap to prevent instability
    N_total     = len(y_train)
    weights     = np.zeros(NUM_CLASSES, dtype=np.float32)

    unique, counts = np.unique(y_train, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))

    print(f"\n[weights] Class weights for weighted cross-entropy:")
    print(f"          {'Lbl':<5} {'Class Name':<22} "
          f"{'Count':>8} {'Weight':>10}")
    print(f"          {'-'*5} {'-'*22} {'-'*8} {'-'*10}")

    for c in range(NUM_CLASSES):
        n_c = count_map.get(c, 0)
        if n_c == 0:
            # Class absent from train — assign max weight
            w = MAX_WEIGHT
        else:
            w = N_total / (NUM_CLASSES * n_c)
            w = min(w, MAX_WEIGHT)   # cap

        weights[c] = w
        name = CLASS_NAMES.get(c, f"Class-{c}")
        print(f"          {c:<5} {name:<22} "
              f"{n_c:>8,} {w:>10.2f}")

        # class_weights = torch.FloatTensor(weights).to(device)
        # print(f"\n          Min weight : {weights.min():.4f}  "
        #     f"(Benign — most frequent)")
        # print(f"          Max weight : {weights.max():.4f}  "
        #     f"(capped at {MAX_WEIGHT})")
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"\n          Min weight : {weights.min():.4f}  "
          f"(Benign — most frequent)")
    print(f"          Max weight : {weights.max():.4f}  "
          f"(capped at {MAX_WEIGHT})")

    return class_weights

# =============================================================
#  4. DATALOADER CREATION
# =============================================================

def create_dataloaders(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val:   np.ndarray,
                       y_val:   np.ndarray,
                       X_test:  np.ndarray,
                       y_test:  np.ndarray,
                       batch_size: int = BATCH_SIZE) -> tuple:
    """
    Create PyTorch DataLoaders for all three splits.

    Train loader : shuffle=True  (randomize batch order)
    Val loader   : shuffle=False (deterministic evaluation)
    Test loader  : shuffle=False (deterministic evaluation)

    num_workers=0 for Mac MPS compatibility.
    On Linux cluster: increase num_workers to 4 for speedup.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader objects
    train_dataset, val_dataset, test_dataset : Dataset objects
    """
    train_dataset = KubernetesSequenceDataset(X_train, y_train)
    val_dataset   = KubernetesSequenceDataset(X_val,   y_val)
    test_dataset  = KubernetesSequenceDataset(X_test,  y_test)

    # num_workers=0 required for MPS on Mac
    # [CLUSTER] change to num_workers=4 for faster loading
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = False,   # pin_memory not supported on MPS
        drop_last   = False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )

    print(f"\n[loader] DataLoaders created.")
    print(f"         Batch size   : {batch_size}")
    print(f"         Train batches: {len(train_loader)}")
    print(f"         Val batches  : {len(val_loader)}")
    print(f"         Test batches : {len(test_loader)}")

    return (train_loader, val_loader, test_loader,
            train_dataset, val_dataset, test_dataset)

# =============================================================
#  5. SANITY CHECK
# =============================================================

def sanity_check(train_loader: DataLoader,
                 device: torch.device):
    """
    Load one batch and verify shapes and types.
    Confirms the DataLoader → model pipeline is correctly wired.
    """
    print(f"\n[sanity] Loading one batch from train_loader ...")
    batch_X, batch_y = next(iter(train_loader))

    print(f"         batch_X shape  : {batch_X.shape}")
    print(f"         batch_y shape  : {batch_y.shape}")
    print(f"         batch_X dtype  : {batch_X.dtype}")
    print(f"         batch_y dtype  : {batch_y.dtype}")
    print(f"         batch_X min    : {batch_X.min():.4f}")
    print(f"         batch_X max    : {batch_X.max():.4f}")
    print(f"         Unique labels  : "
          f"{sorted(batch_y.unique().tolist())}")

    # Move to device and verify
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    print(f"         Device test    : "
          f"X on {batch_X.device}, y on {batch_y.device} ✓")

    # Expected: batch_X = (batch_size, W, D) = (64, 16, 39)
    assert batch_X.dim() == 3, \
        f"Expected 3D tensor (B, W, D), got {batch_X.dim()}D"
    assert batch_X.shape[1] == 16, \
        f"Expected W=16, got {batch_X.shape[1]}"
    assert batch_X.shape[2] == 39, \
        f"Expected D=39, got {batch_X.shape[2]}"

    print(f"\n[sanity] ✓ All checks passed.")
    print(f"         Tensor shape (B, W, D) = {tuple(batch_X.shape)}")
    print(f"         Ready to feed into transformer encoder.")

# =============================================================
#  6. MAIN PIPELINE
# =============================================================

def run_stage4() -> tuple:
    """
    Execute complete Stage 4.

    Loads numpy arrays from DATA_DIR (saved by stage3).
    Returns everything needed for model training in Stage 6.

    Returns
    -------
    train_loader   : DataLoader
    val_loader     : DataLoader
    test_loader    : DataLoader
    class_weights  : torch.Tensor  (NUM_CLASSES,) on device
    device         : torch.device
    datasets       : dict with train/val/test Dataset objects
    """
    print("\n" + "=" * 62)
    print("  STAGE 4 — DATALOADER CONSTRUCTION")
    print("=" * 62 + "\n")

    # ── Detect device ─────────────────────────────────────────
    device = get_device()

    # ── Load numpy arrays ─────────────────────────────────────
    print(f"\n[load]  Loading split arrays from {DATA_DIR} ...")
    arrays = {}
    for name in ["X_train", "y_train", "X_val",
                 "y_val", "X_test", "y_test"]:
        path = os.path.join(DATA_DIR, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name}.npy not found at {path}\n"
                f"Run stage3_sequences.py first."
            )
        arrays[name] = np.load(path)
        print(f"        {name:<10} → {arrays[name].shape}")

    # ── Compute class weights ─────────────────────────────────
    class_weights = compute_class_weights(
        arrays["y_train"], device
    )

    # ── Create DataLoaders ────────────────────────────────────
    (train_loader, val_loader, test_loader,
     train_ds, val_ds, test_ds) = create_dataloaders(
        arrays["X_train"], arrays["y_train"],
        arrays["X_val"],   arrays["y_val"],
        arrays["X_test"],  arrays["y_test"],
        batch_size = BATCH_SIZE
    )

    # ── Sanity check ──────────────────────────────────────────
    sanity_check(train_loader, device)

    datasets = {
        "train" : train_ds,
        "val"   : val_ds,
        "test"  : test_ds,
    }

    print(f"\n[stage4] ✓ Complete.")
    print(f"         Device        : {device}")
    print(f"         Train loader  : {len(train_loader)} batches")
    print(f"         Val loader    : {len(val_loader)} batches")
    print(f"         Test loader   : {len(test_loader)} batches")
    print(f"         Class weights : shape {class_weights.shape}")
    print(f"         → Pass to stage5_model.py and "
          f"stage6_train.py\n")

    return (train_loader, val_loader, test_loader,
            class_weights, device, datasets)


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  Terminal:
#    python scripts/stage4_dataloader.py
#
#  After stage3 in notebook:
#    from scripts.stage4_dataloader import run_stage4
#    train_loader, val_loader, test_loader, \
#        class_weights, device, datasets = run_stage4()
#
# =============================================================

if __name__ == "__main__":
    run_stage4()