# =============================================================
#  stage3_sequences.py
#  Stage 3 — Temporal Sequence Construction and Dataset Splitting
#
#  This stage implements Novelty 1 of the paper:
#  "Sequence-Aware Misuse Detection in Kubernetes Using
#   Transformer Models"
#
#  What this script does:
#    1. Sorts flows temporally (by row order — CICFlowMeter
#       output is chronological; timestamp column not available
#       after preprocessing)
#    2. Constructs sliding window sequences (W=16, S=8)
#    3. Assigns majority-vote labels with attack-priority tie-break
#    4. Performs minimum-guarantee stratified split:
#         - For classes with < MIN_SPLIT_GUARANTEE sequences,
#           at least 1 sequence is reserved for val and test
#         - Remaining sequences split 70/15/15 stratified
#    5. Re-fits StandardScaler on train split only (no leakage)
#    6. Saves all splits as numpy arrays
#    7. Reports full sequence statistics and per-split distributions
#
#  Key design decision — minimum-guarantee split:
#    Standard stratified splitting with 15% test ratio produces
#    0 sequences in some splits for classes with only 3-7 sequences.
#    The minimum-guarantee strategy ensures every class that exists
#    in the dataset is represented in all three splits.
#    This is documented in paper Section 4.3.3.
#
#  Input  : (df_norm, feature_cols) from stage2
#  Output : (X_train, y_train, X_val, y_val,
#             X_test,  y_test,  scaler_train, feature_cols)
# =============================================================

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
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
LABEL_COLUMN = project_config.LABEL_COLUMN
TIMESTAMP_COL = project_config.TIMESTAMP_COL
WINDOW_SIZE = project_config.WINDOW_SIZE
STRIDE = project_config.STRIDE
TIE_BREAK = project_config.TIE_BREAK
TRAIN_RATIO = project_config.TRAIN_RATIO
VAL_RATIO = project_config.VAL_RATIO
TEST_RATIO = project_config.TEST_RATIO
RANDOM_SEED = project_config.RANDOM_SEED
CLASS_NAMES = project_config.CLASS_NAMES
CLASS_SHORT = project_config.CLASS_SHORT
NUM_CLASSES = project_config.NUM_CLASSES

# Minimum sequences guaranteed in val and test for rare classes
MIN_SPLIT_GUARANTEE = 1

# =============================================================
#  1. TEMPORAL SORTING
# =============================================================

def sort_temporally(df: pd.DataFrame,
                    feature_cols: list) -> pd.DataFrame:
    """
    Sort flows by timestamp if available.
    If not, preserve existing row order.

    Note: The timestamp column was dropped as a non-numeric
    string column in Stage 2. CICFlowMeter output is
    approximately chronological by row order, so we preserve
    the existing order. This is documented in the paper.
    """
    if TIMESTAMP_COL in df.columns:
        try:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
            df_sorted = df.sort_values(
                TIMESTAMP_COL
            ).reset_index(drop=True)
            print(f"[sort]  Sorted by '{TIMESTAMP_COL}' column.")
        except Exception:
            df_sorted = df.reset_index(drop=True)
            print(f"[sort]  Timestamp parse failed.")
            print(f"        Preserving CICFlowMeter row order "
                  f"(approximately chronological).")
    else:
        df_sorted = df.reset_index(drop=True)
        print(f"[sort]  Timestamp column not available after "
              f"preprocessing.")
        print(f"        Using CICFlowMeter row order "
              f"(approximately chronological).")
        print(f"        This is documented in paper Section 4.3.1.")

    print(f"        Total flows: {len(df_sorted):,}")
    return df_sorted

# =============================================================
#  2. SLIDING WINDOW SEQUENCE CONSTRUCTION
# =============================================================

def build_sequences(df: pd.DataFrame,
                    feature_cols: list,
                    window_size: int = WINDOW_SIZE,
                    stride: int = STRIDE) -> tuple:
    """
    Construct sequences via sliding window over ordered flows.

    Each sequence:
      Shape : (window_size, D)
      Label : majority vote with attack-priority tie-breaking

    Returns
    -------
    X : np.ndarray  (N_sequences, W, D)
    y : np.ndarray  (N_sequences,)
    """
    X_list   = []
    y_list   = []
    values   = df[feature_cols].values   # (n_flows, D)
    labels   = df[LABEL_COLUMN].values   # (n_flows,)
    n_flows  = len(df)

    i = 0
    while i + window_size <= n_flows:
        window_X      = values[i : i + window_size]
        window_labels = labels[i : i + window_size]
        label_counts  = Counter(window_labels.tolist())
        majority_label = _majority_vote(label_counts)

        X_list.append(window_X)
        y_list.append(majority_label)
        i += stride

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int64)

    print(f"\n[sequences] Sliding window complete.")
    print(f"            W={window_size}, S={stride}")
    print(f"            Input flows     : {n_flows:,}")
    print(f"            Sequences built : {len(X):,}")
    print(f"            Shape           : {X.shape}")

    return X, y


def _majority_vote(label_counts: Counter) -> int:
    """
    Majority vote with attack-priority tie-breaking.
    Attack class wins over benign on a tie.
    If two attack classes tie, higher label index wins
    (deterministic).
    """
    max_count  = max(label_counts.values())
    candidates = [
        lbl for lbl, cnt in label_counts.items()
        if cnt == max_count
    ]
    if len(candidates) == 1:
        return int(candidates[0])

    if TIE_BREAK == "attack":
        attack_cands = [c for c in candidates if c > 0]
        if attack_cands:
            return int(max(attack_cands))
        return int(candidates[0])
    else:
        if 0 in candidates:
            return 0
        return int(candidates[0])

# =============================================================
#  3. MINIMUM-GUARANTEE STRATIFIED SPLIT
# =============================================================

def split_sequences(X: np.ndarray,
                    y: np.ndarray) -> tuple:
    """
    Split sequences into train / val / test with a
    minimum-guarantee strategy for rare classes.

    Strategy:
      For each class with < MIN_SPLIT_GUARANTEE*2 sequences
      available for val+test, we manually reserve 1 sequence
      for val and 1 for test before the random split.
      Remaining sequences are split 70/15/15 stratified.

    This ensures every class present in the full dataset
    appears in all three splits — critical for computing
    per-class metrics on all 11 classes during evaluation.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    rng = np.random.RandomState(RANDOM_SEED)

    # ── Group indices by class ────────────────────────────────
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[int(label)].append(idx)

    train_idx = []
    val_idx   = []
    test_idx  = []

    print(f"\n[split]  Minimum-guarantee stratified split:")
    print(f"         Guarantee: {MIN_SPLIT_GUARANTEE} seq "
          f"per class in val and test")
    print(f"\n         {'Class':<22} {'Total':>6} "
          f"{'Train':>6} {'Val':>5} {'Test':>5}  Note")
    print(f"         {'-'*22} {'-'*6} {'-'*6} {'-'*5} {'-'*5}  {'-'*20}")

    for class_label in sorted(class_indices.keys()):
        indices = class_indices[class_label]
        n       = len(indices)
        name    = CLASS_NAMES.get(class_label, str(class_label))

        # Shuffle indices for this class
        shuffled = rng.permutation(indices).tolist()

        # Minimum needed for val and test
        min_needed = MIN_SPLIT_GUARANTEE * 2

        if n <= min_needed + 1:
            # Very rare class — manually assign
            # Reserve 1 for val, 1 for test, rest for train
            guaranteed_val  = shuffled[:MIN_SPLIT_GUARANTEE]
            guaranteed_test = shuffled[MIN_SPLIT_GUARANTEE:
                                       MIN_SPLIT_GUARANTEE * 2]
            remaining_train = shuffled[MIN_SPLIT_GUARANTEE * 2:]

            val_idx.extend(guaranteed_val)
            test_idx.extend(guaranteed_test)
            train_idx.extend(remaining_train)

            note = "min-guarantee applied"
            print(f"         {name:<22} {n:>6} "
                  f"{len(remaining_train):>6} "
                  f"{len(guaranteed_val):>5} "
                  f"{len(guaranteed_test):>5}  {note}")

        else:
            # Normal class — proportional stratified split
            n_val   = max(MIN_SPLIT_GUARANTEE,
                          int(round(n * VAL_RATIO)))
            n_test  = max(MIN_SPLIT_GUARANTEE,
                          int(round(n * TEST_RATIO)))
            n_train = n - n_val - n_test

            if n_train < 1:
                # Edge case — ensure at least 1 train sample
                n_train = 1
                n_val   = max(1, (n - 1) // 2)
                n_test  = n - 1 - n_val

            val_idx.extend(shuffled[:n_val])
            test_idx.extend(shuffled[n_val : n_val + n_test])
            train_idx.extend(shuffled[n_val + n_test:])

            note = "stratified"
            print(f"         {name:<22} {n:>6} "
                  f"{n_train:>6} {n_val:>5} {n_test:>5}  {note}")

    # ── Build split arrays ────────────────────────────────────
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val   = X[val_idx]
    y_val   = y[val_idx]
    X_test  = X[test_idx]
    y_test  = y[test_idx]

    # ── Shuffle train set ─────────────────────────────────────
    train_perm = rng.permutation(len(X_train))
    X_train    = X_train[train_perm]
    y_train    = y_train[train_perm]

    total = len(X_train) + len(X_val) + len(X_test)
    print(f"\n         Train : {len(X_train):,}  "
          f"({len(X_train)/total*100:.1f}%)")
    print(f"         Val   : {len(X_val):,}  "
          f"({len(X_val)/total*100:.1f}%)")
    print(f"         Test  : {len(X_test):,}  "
          f"({len(X_test)/total*100:.1f}%)")
    print(f"         Total : {total:,}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# =============================================================
#  4. SEQUENCE DISTRIBUTION REPORT
# =============================================================

def print_sequence_distribution(y: np.ndarray,
                                 split_name: str = "full"):
    """Print class distribution of a sequence array."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    print(f"\n  Sequence distribution — {split_name}:")
    print(f"  {'Lbl':<5} {'Class Name':<22} "
          f"{'Count':>8} {'Pct':>7}")
    print(f"  {'-'*5} {'-'*22} {'-'*8} {'-'*7}")

    all_labels = set(range(NUM_CLASSES))
    found      = set(int(l) for l in unique)
    missing    = all_labels - found

    for lbl, cnt in zip(unique, counts):
        name = CLASS_NAMES.get(int(lbl), f"Unknown-{lbl}")
        pct  = int(cnt) / total * 100
        print(f"  {int(lbl):<5} {name:<22} "
              f"{int(cnt):>8,} {pct:>6.2f}%")

    if missing:
        for lbl in sorted(missing):
            name = CLASS_NAMES.get(lbl, f"Unknown-{lbl}")
            print(f"  {lbl:<5} {name:<22} "
                  f"{'0':>8}  {'0.00%':>7}  ← absent")

    print(f"  {'TOTAL':<28} {total:>8,}")

# =============================================================
#  5. RE-FIT SCALER ON TRAIN ONLY
# =============================================================

def refit_scaler_on_train(X_train, X_val, X_test) -> tuple:
    """
    Re-fit StandardScaler on training sequences ONLY.
    Apply fitted scaler to val and test.
    Prevents data leakage from val/test into normalization.
    """
    D = X_train.shape[2]

    X_train_flat = X_train.reshape(-1, D)
    X_val_flat   = X_val.reshape(-1, D)
    X_test_flat  = X_test.reshape(-1, D)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train_flat).reshape(X_train.shape)
    X_val_scaled   = scaler.transform(
        X_val_flat).reshape(X_val.shape)
    X_test_scaled  = scaler.transform(
        X_test_flat).reshape(X_test.shape)

    print(f"\n[scaler] Re-fit on train split only — no leakage.")
    print(f"         Train mean range : "
          f"[{X_train_scaled.reshape(-1,D).mean(axis=0).min():.4f}, "
          f"{X_train_scaled.reshape(-1,D).mean(axis=0).max():.4f}]")
    print(f"         Train std range  : "
          f"[{X_train_scaled.reshape(-1,D).std(axis=0).min():.4f}, "
          f"{X_train_scaled.reshape(-1,D).std(axis=0).max():.4f}]")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# =============================================================
#  6. SPLIT DISTRIBUTION PLOT
# =============================================================

def plot_split_distributions(y_train, y_val, y_test,
                              save: bool = True):
    """Side-by-side bar chart of class distributions per split."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    splits = [
        (y_train, "Train",      "#378ADD"),
        (y_val,   "Validation", "#1D9E75"),
        (y_test,  "Test",       "#E24B4A"),
    ]

    for ax, (y_split, title, color) in zip(axes, splits):
        unique, counts = np.unique(y_split, return_counts=True)
        labels = [CLASS_SHORT.get(int(i), str(i))
                  for i in range(NUM_CLASSES)
                  if int(i) in unique]
        cnt_map = dict(zip(unique.tolist(), counts.tolist()))
        bar_counts = [cnt_map.get(i, 0)
                      for i in range(NUM_CLASSES)
                      if i in cnt_map]

        ax.bar(labels, bar_counts, color=color,
               edgecolor="white", linewidth=0.5)
        ax.set_title(f"{title}  (n={len(y_split):,})",
                     fontsize=11)
        ax.set_ylabel("Sequences", fontsize=9)
        ax.tick_params(axis="x", rotation=40, labelsize=7.5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle(
        f"Sequence class distribution per split  "
        f"(W={WINDOW_SIZE}, S={STRIDE})",
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR,
                           "split_distributions.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot]  Split distributions → {out}")
    plt.show()
    plt.close()

# =============================================================
#  7. SAVE SPLITS
# =============================================================

def save_splits(X_train, y_train, X_val, y_val,
                X_test, y_test, scaler_train, feature_cols):
    """Save all numpy arrays and metadata to DATA_DIR."""
    arrays = {
        "X_train": X_train, "y_train": y_train,
        "X_val"  : X_val,   "y_val"  : y_val,
        "X_test" : X_test,  "y_test" : y_test,
    }
    for name, arr in arrays.items():
        path = os.path.join(DATA_DIR, f"{name}.npy")
        np.save(path, arr)
        print(f"[save]  {name:<10} → shape {arr.shape}  {path}")

    scaler_path = os.path.join(DATA_DIR, "scaler_train.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_train, f)
    print(f"[save]  scaler_train → {scaler_path}")

    feat_path = os.path.join(DATA_DIR, "feature_names.pkl")
    with open(feat_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"[save]  feature_names ({len(feature_cols)}) → {feat_path}")

    meta = {
        "window_size"  : WINDOW_SIZE,
        "stride"       : STRIDE,
        "n_features"   : len(feature_cols),
        "n_classes"    : NUM_CLASSES,
        "train_shape"  : X_train.shape,
        "val_shape"    : X_val.shape,
        "test_shape"   : X_test.shape,
        "feature_names": feature_cols,
    }
    meta_path = os.path.join(DATA_DIR, "split_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[save]  metadata     → {meta_path}")

# =============================================================
#  8. MAIN PIPELINE
# =============================================================

def run_stage3(df_norm: pd.DataFrame,
               feature_cols: list,
               scaler=None) -> tuple:
    """
    Execute complete Stage 3.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    scaler_train, feature_cols
    """
    print("\n" + "=" * 62)
    print("  STAGE 3 — SEQUENCE CONSTRUCTION AND SPLITTING")
    print("=" * 62 + "\n")

    print(f"  Window size W  : {WINDOW_SIZE}")
    print(f"  Stride S       : {STRIDE}")
    print(f"  Tie-break      : {TIE_BREAK}")
    print(f"  Features D     : {len(feature_cols)}")
    print(f"  Input flows    : {len(df_norm):,}\n")

    # Step 1 — Temporal sort
    df_sorted = sort_temporally(df_norm, feature_cols)

    # Step 2 — Build sequences
    X, y = build_sequences(
        df_sorted, feature_cols, WINDOW_SIZE, STRIDE
    )[:2]

    # Step 3 — Full distribution
    print_sequence_distribution(y, "full dataset")

    # Step 4 — Minimum-guarantee split
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test) = split_sequences(X, y)

    # Step 5 — Per-split distributions
    print_sequence_distribution(y_train, "train")
    print_sequence_distribution(y_val,   "validation")
    print_sequence_distribution(y_test,  "test")

    # Step 6 — Re-fit scaler on train only
    (X_train, X_val, X_test,
     scaler_train) = refit_scaler_on_train(
        X_train, X_val, X_test
    )

    # Step 7 — Plot
    plot_split_distributions(y_train, y_val, y_test, save=True)

    # Step 8 — Save
    print(f"\n[save]  Saving splits to {DATA_DIR}:")
    save_splits(
        X_train, y_train, X_val, y_val,
        X_test,  y_test,  scaler_train, feature_cols
    )

    print(f"\n[stage3] ✓ Complete.")
    print(f"         X_train : {X_train.shape}")
    print(f"         X_val   : {X_val.shape}")
    print(f"         X_test  : {X_test.shape}")
    print(f"         Input dim D = {X_train.shape[2]}")
    print(f"         → Pass to stage4_dataloader.run_stage4()\n")

    return (X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            scaler_train,
            feature_cols)


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  Terminal:
#    python scripts/stage3_sequences.py
#
#  Notebook after stage2:
#    from scripts.stage3_sequences import run_stage3
#    X_train, y_train, X_val, y_val, X_test, y_test, \
#        scaler_train, feature_cols = run_stage3(df_norm, feature_cols)
#
# =============================================================

if __name__ == "__main__":
    pre_path  = os.path.join(DATA_DIR, "preprocessed.csv")
    feat_path = os.path.join(DATA_DIR, "feature_names.pkl")

    if not os.path.exists(pre_path):
        raise FileNotFoundError(
            f"preprocessed.csv not found.\n"
            f"Run stage2_preprocessing.py first."
        )
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"feature_names.pkl not found.\n"
            f"Run stage2_preprocessing.py first."
        )

    print("[load]  Reading preprocessed.csv ...")
    df_norm = pd.read_csv(pre_path, low_memory=False)
    print(f"        Shape: {df_norm.shape}")

    with open(feat_path, "rb") as f:
        feature_cols = pickle.load(f)
    print(f"[load]  Features: {len(feature_cols)}")

    run_stage3(df_norm, feature_cols)