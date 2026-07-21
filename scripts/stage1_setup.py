# =============================================================
#  stage1_setup.py
#  Stage 1 — Environment setup, data loading, inspection.
#
#  PRIMARY: VS Code on Mac (local environment)
#  Run as script   : python scripts/stage1_setup.py
#  Run in notebook : from scripts.stage1_setup import run_stage1
#
#  What this script does:
#    1. Sets random seeds for reproducibility
#    2. Creates all project directories
#    3. Verifies dataset CSV files exist
#    4. Loads and merges benign.csv + malicious.csv
#    5. Normalizes column names
#    6. Prints full inspection report
#    7. Plots and saves class distribution chart
#    8. Saves merged raw CSV
#
#  Output: (df, summary) tuple passed to stage2_preprocessing.py
# =============================================================

import os
import sys
import random
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import warnings


warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 120

# ── Path setup ────────────────────────────────────────────────
# Add project root to sys.path so config.py is always findable
# regardless of where the script is called from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# [KAGGLE] replace PROJECT_ROOT line above with:
# PROJECT_ROOT = "/kaggle/working"

# [CLUSTER] replace PROJECT_ROOT line above with:
# PROJECT_ROOT = "/home/<your_username>/kubernetes_anomaly_detection"

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.py")
spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load config module from: {CONFIG_PATH}")
project_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_config)

ENVIRONMENT = project_config.ENVIRONMENT
BASE_DIR = project_config.BASE_DIR
DATA_DIR = project_config.DATA_DIR
CHECKPOINT_DIR = project_config.CHECKPOINT_DIR
RESULTS_DIR = project_config.RESULTS_DIR
LOG_DIR = project_config.LOG_DIR
BENIGN_CSV = project_config.BENIGN_CSV
MALICIOUS_CSV = project_config.MALICIOUS_CSV
RANDOM_SEED = project_config.RANDOM_SEED
CLASS_NAMES = project_config.CLASS_NAMES
CLASS_SHORT = project_config.CLASS_SHORT
LABEL_COLUMN = project_config.LABEL_COLUMN
NUM_CLASSES = project_config.NUM_CLASSES

# =============================================================
#  1. REPRODUCIBILITY
# =============================================================

def set_seed(seed: int = RANDOM_SEED):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed]  Random seed fixed → {seed}")

# =============================================================
#  2. DIRECTORY CREATION
# =============================================================

def create_dirs():
    """Create all project directories if they do not exist."""
    dirs = [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[dirs]  Project directories ready:")
    for d in dirs:
        print(f"        {d}")

# =============================================================
#  3. DATA AVAILABILITY CHECK
# =============================================================

def check_data_available() -> bool:
    """
    Verify both CSV files exist at paths defined in config.py.
    If not found, prints a helpful diagnostic and returns False.
    """
    print(f"\n[data]  Checking CSV paths:")
    print(f"        BENIGN_CSV    → {BENIGN_CSV}")
    print(f"        MALICIOUS_CSV → {MALICIOUS_CSV}")

    benign_ok    = os.path.exists(BENIGN_CSV)
    malicious_ok = os.path.exists(MALICIOUS_CSV)

    if benign_ok and malicious_ok:
        size_b = os.path.getsize(BENIGN_CSV)    / (1024 ** 2)
        size_m = os.path.getsize(MALICIOUS_CSV) / (1024 ** 2)
        print(f"[data]  benign.csv    ✓  ({size_b:.1f} MB)")
        print(f"[data]  malicious.csv ✓  ({size_m:.1f} MB)")
        return True

    # ── Diagnostic: show what is in data folder ───────────────
    print("\n[data]  ERROR: One or both CSV files not found.")
    print(f"[data]  Looking inside: {DATA_DIR}\n")

    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        if files:
            for f in files:
                fpath = os.path.join(DATA_DIR, f)
                size  = os.path.getsize(fpath) / (1024 ** 2)
                print(f"        {f}  ({size:.2f} MB)")
        else:
            print("        (folder is empty)")
    else:
        print("        data/ folder does not exist yet.")

    print("\n[data]  To fix: download the dataset and place")
    print(f"        benign.csv and malicious.csv inside:")
    print(f"        {DATA_DIR}")
    print("\n[data]  Download command (run in terminal with venv active):")
    print(f"        cd {DATA_DIR}")
    print(f"        kaggle datasets download dinesh6627/sever-and-dogan-2023-dataset")
    print(f"        unzip sever-and-dogan-2023-dataset.zip")
    print(f"        rm sever-and-dogan-2023-dataset.zip")
    return False

# =============================================================
#  4. LOAD AND MERGE
# =============================================================

def load_data() -> pd.DataFrame:
    """
    Load benign.csv and malicious.csv, assign labels,
    merge into one unified DataFrame.

    Label scheme (Sever & Dogan, 2023):
      benign.csv    → label = 0  (forced)
      malicious.csv → labels 1-10 (already present)
    """
    print("\n[load]  Reading benign.csv ...")
    benign = pd.read_csv(BENIGN_CSV, low_memory=False)
    print(f"        Shape: {benign.shape}")

    print("[load]  Reading malicious.csv ...")
    malicious = pd.read_csv(MALICIOUS_CSV, low_memory=False)
    print(f"        Shape: {malicious.shape}")

    # Force benign label = 0 regardless of what column exists
    benign[LABEL_COLUMN] = 0

    # Handle capitalization variants in malicious label column
    label_variants = [LABEL_COLUMN, "Label", "LABEL", "label"]
    found = next(
        (v for v in label_variants if v in malicious.columns),
        None
    )
    if found is None:
        raise ValueError(
            f"Label column not found in malicious.csv.\n"
            f"Tried: {label_variants}\n"
            f"Available columns: {list(malicious.columns[:10])}"
        )
    if found != LABEL_COLUMN:
        malicious = malicious.rename(columns={found: LABEL_COLUMN})

    # Merge
    df = pd.concat([benign, malicious], ignore_index=True)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    print(f"\n[merge] Complete.")
    print(f"        Benign rows    : {len(benign):>12,}")
    print(f"        Malicious rows : {len(malicious):>12,}")
    print(f"        Total rows     : {len(df):>12,}")
    print(f"        Total columns  : {df.shape[1]}")
    return df

# =============================================================
#  5. NORMALIZE COLUMN NAMES
# =============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize all column names:
      strip whitespace → replace spaces with _ → lowercase
    Example: 'Flow Duration' → 'flow_duration'
             'Label'         → 'label'
    """
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(" ", "_", regex=False)
          .str.replace(r"[^\w]", "_", regex=True)
          .str.lower()
    )

    # Resolve duplicate "label" columns created after normalization
    # (e.g., original "Label" and "label" both present).
    label_count = int((df.columns == LABEL_COLUMN).sum())
    if label_count > 1:
        label_candidates = df.loc[:, df.columns == LABEL_COLUMN]
        df[LABEL_COLUMN] = label_candidates.bfill(axis=1).iloc[:, 0]

        keep_mask = []
        seen_label = False
        for col in df.columns:
            if col == LABEL_COLUMN:
                if seen_label:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    seen_label = True
            else:
                keep_mask.append(True)
        df = df.loc[:, keep_mask]

    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    if df[LABEL_COLUMN].isna().any():
        raise ValueError("Label column contains NaN after normalization.")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    print(f"\n[cols]  Column names normalized.")
    print(f"        Label column  : '{LABEL_COLUMN}'")
    print(f"        Total columns : {df.shape[1]}")
    print(f"        First 6 cols  : {list(df.columns[:6])}")
    return df

# =============================================================
#  6. DATASET INSPECTION REPORT
# =============================================================

def inspect_data(df: pd.DataFrame) -> dict:
    """
    Print structured inspection report covering:
      - Shape and dtype summary
      - NaN counts per column
      - Inf counts per column
      - Class distribution with imbalance ratio
    Returns summary dict for downstream stages.
    """
    sep = "=" * 62
    print(f"\n{sep}")
    print("  DATASET INSPECTION REPORT")
    print(sep)

    # Shape
    print(f"\n  Shape : {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # Dtypes
    dtype_counts = df.dtypes.value_counts()
    print(f"\n  Dtypes:")
    for dtype, count in dtype_counts.items():
        print(f"    {str(dtype):<15}  {int(count):>4} columns")

    # NaN
    nan_counts = df.isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    nan_pct    = (nan_counts / len(df) * 100).round(2)

    print(f"\n  NaN values : {len(nan_cols)} columns affected")
    if len(nan_cols) > 0:
        print(f"  {'Column':<45} {'Count':>8}  {'Pct':>6}")
        print(f"  {'-'*45} {'-'*8}  {'-'*6}")
        for i, (col, count_val_raw) in enumerate(nan_cols.iloc[:15].items()):
            count_val = int(count_val_raw)
            pct_val   = float(nan_pct.iloc[i])
            print(f"  {col:<45} {count_val:>8,}  {pct_val:>5.1f}%")
        if len(nan_cols) > 15:
            print(f"  ... and {len(nan_cols) - 15} more columns")

    # Inf
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = [
        int(np.isinf(numeric_df.iloc[:, i].to_numpy()).sum())
        for i in range(numeric_df.shape[1])
    ]
    inf_series   = pd.Series(inf_counts, index=numeric_df.columns)
    inf_cols     = inf_series[inf_series > 0]

    print(f"\n  Inf values : {len(inf_cols)} columns affected")
    if len(inf_cols) > 0:
        print(f"  {'Column':<45} {'Count':>8}")
        print(f"  {'-'*45} {'-'*8}")
        for col, count_val_raw in inf_cols.iloc[:15].items():
            count_val = int(count_val_raw)
            print(f"  {col:<45} {count_val:>8,}")
        if len(inf_cols) > 15:
            print(f"  ... and {len(inf_cols) - 15} more columns")

    # Class distribution
    label_data = df[LABEL_COLUMN]
    if isinstance(label_data, pd.DataFrame):
        label_series = label_data.bfill(axis=1).iloc[:, 0]
    else:
        label_series = label_data

    class_counts    = label_series.value_counts().sort_index()
    total           = len(df)
    majority_count  = int(class_counts.max())
    minority_count  = int(class_counts.min())
    imbalance_ratio = majority_count / minority_count

    print(f"\n  Class Distribution ({NUM_CLASSES} classes):")
    print(f"  {'Lbl':<5} {'Class Name':<22} "
          f"{'Count':>10} {'Pct':>7}  Bar")
    print(f"  {'-'*5} {'-'*22} {'-'*10} {'-'*7}  {'-'*20}")
    for label, count in class_counts.items():
        name  = CLASS_NAMES.get(int(label), f"Unknown-{label}")
        cnt   = int(count)
        pct   = cnt / total * 100
        bar   = "█" * int(pct / 2)
        print(f"  {int(label):<5} {name:<22} "
              f"{cnt:>10,} {pct:>6.2f}%  {bar}")

    print(f"\n  Majority : {majority_count:,}")
    print(f"  Minority : {minority_count:,}")
    print(f"  Imbalance: {imbalance_ratio:.1f} : 1")
    print(f"\n{sep}\n")

    return {
        "n_rows"          : df.shape[0],
        "n_cols"          : df.shape[1],
        "nan_cols"        : {str(k): int(v)
                             for k, v in nan_cols.items()},
        "inf_cols"        : {str(k): int(v)
                             for k, v in inf_cols.items()},
        "class_counts"    : {int(k): int(v)
                             for k, v in class_counts.items()},
        "imbalance_ratio" : round(imbalance_ratio, 2),
    }

# =============================================================
#  7. CLASS DISTRIBUTION PLOT
# =============================================================

def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    """
    Bar chart of per-class flow counts.
    Benign = green, attack classes = red.
    Saves to results/class_distribution.png
    """
    label_data = df[LABEL_COLUMN]
    if isinstance(label_data, pd.DataFrame):
        label_series = label_data.bfill(axis=1).iloc[:, 0]
    else:
        label_series = label_data

    class_counts = label_series.value_counts().sort_index()
    labels  = [CLASS_SHORT.get(int(i), str(i))
               for i in class_counts.index]
    counts  = [int(c) for c in class_counts.values]
    colors  = ["#1D9E75" if int(i) == 0 else "#E24B4A"
               for i in class_counts.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(labels, counts, color=colors,
                  edgecolor="white", linewidth=0.6)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.008,
            f"{count:,}",
            ha="center", va="bottom",
            fontsize=7.5, color="#333333"
        )

    ax.set_title(
        "Class distribution — Sever & Dogan (2023) Dataset\n"
        "Kubernetes Misuse Detection",
        fontsize=12, pad=10
    )
    ax.set_ylabel("Number of network flows", fontsize=10)
    ax.set_xlabel("Attack class", fontsize=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(handles=[
        Patch(facecolor="#1D9E75", edgecolor="none",
              label="Benign (class 0)"),
        Patch(facecolor="#E24B4A", edgecolor="none",
              label="Attack classes (1-10)"),
    ], fontsize=9, framealpha=0.7)

    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "class_distribution.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot]  Saved → {out}")

    plt.show()
    plt.close()

# =============================================================
#  8. MAIN PIPELINE
# =============================================================

def run_stage1() -> tuple:
    """
    Execute complete Stage 1.

    Returns
    -------
    df      : pd.DataFrame  — raw merged dataset
    summary : dict          — inspection statistics
    """
    print("\n" + "=" * 62)
    print("  STAGE 1 — SETUP AND DATA LOADING")
    print("=" * 62 + "\n")

    set_seed()
    create_dirs()

    if not check_data_available():
        raise RuntimeError(
            "Dataset files not found. "
            "Follow the download instructions printed above."
        )

    df      = load_data()
    df      = normalize_columns(df)
    summary = inspect_data(df)
    plot_class_distribution(df, save=True)

    # Save merged raw CSV once for reproducibility
    raw_out = os.path.join(DATA_DIR, "merged_raw.csv")
    df.to_csv(raw_out, index=False)
    print(f"[save]  Merged raw CSV → {raw_out}")

    print(f"\n[stage1] ✓ Complete.")
    print(f"         DataFrame shape  : {df.shape}")
    print(f"         → Pass df to "
          f"stage2_preprocessing.run_stage2(df)\n")

    return df, summary


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  From VS Code terminal (venv activated):
#    cd ~/Documents/kubernetes_anomaly_detection
#    python scripts/stage1_setup.py
#
#  From a Jupyter notebook inside VS Code:
#    import sys
#    sys.path.insert(0, '/Users/<you>/Documents/kubernetes_anomaly_detection')
#    from scripts.stage1_setup import run_stage1
#    df, summary = run_stage1()
#
#  [KAGGLE] sys.path.insert(0, '/kaggle/working')
#  [CLUSTER] sys.path.insert(0, '/home/<username>/kubernetes_anomaly_detection')
#
# =============================================================

if __name__ == "__main__":
    df, summary = run_stage1()