# =============================================================
#  stage2_preprocessing.py
#  Stage 2 — Data Preprocessing and Feature Selection.
#
#  What this script does:
#    1. Identifies and drops non-numeric columns
#    2. Drops the 26 corrupted zero-duration flow rows
#       (NaN in flow_iat_* and Inf in flow_bytes_s, flow_packets_s)
#    3. Drops near-zero variance features
#    4. Drops one column from each highly correlated pair
#    5. Applies z-score normalization (fit on train, apply to all)
#    6. Saves the cleaned feature matrix and label vector
#    7. Reports full feature selection audit trail
#
#  Design decision:
#    No PCA or autoencoders are used. Correlation-based selection
#    preserves feature interpretability and differentiates this
#    work from Aly et al. [32] and Allahabadi [34] who both used
#    PCA and autoencoders on this dataset.
#
#  Input  : df (pd.DataFrame) from stage1_setup.run_stage1()
#  Output : (X, y, feature_names, scaler) tuple
#           passed to stage3_sequences.py
# =============================================================

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import pickle

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
LOG_DIR = project_config.LOG_DIR
LABEL_COLUMN = project_config.LABEL_COLUMN
TIMESTAMP_COL = project_config.TIMESTAMP_COL
NAN_DROP_THRESHOLD = project_config.NAN_DROP_THRESHOLD
VARIANCE_THRESHOLD = project_config.VARIANCE_THRESHOLD
CORRELATION_THRESHOLD = project_config.CORRELATION_THRESHOLD
RANDOM_SEED = project_config.RANDOM_SEED

# =============================================================
#  1. DROP NON-NUMERIC COLUMNS
# =============================================================

def drop_non_numeric(df: pd.DataFrame) -> tuple:
    """
    Identify and drop non-numeric (string/object) columns.
    These are metadata fields — IP addresses, protocol names,
    flags as strings — that cannot be used as model features.

    The label column is preserved separately before dropping.

    Returns
    -------
    df_clean     : pd.DataFrame  — numeric columns only + label
    dropped_cols : list          — names of dropped columns
    """
    # Separate label before any column operations
    y = df[LABEL_COLUMN].copy()

    # Also preserve timestamp if present (needed for Stage 3 sorting)
    has_timestamp = TIMESTAMP_COL in df.columns
    if has_timestamp:
        ts = df[TIMESTAMP_COL].copy()

    # Identify non-numeric columns excluding label and timestamp
    exclude = [LABEL_COLUMN]
    if has_timestamp:
        exclude.append(TIMESTAMP_COL)

    non_numeric = df.drop(columns=exclude).select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    dropped_cols = non_numeric
    df_clean = df.drop(columns=non_numeric)

    print(f"[drop_str]  Non-numeric columns dropped: "
          f"{len(dropped_cols)}")
    for col in dropped_cols:
        print(f"            - {col}")

    return df_clean, dropped_cols

# =============================================================
#  2. DROP CORRUPTED ROWS (NaN and Inf)
# =============================================================

def drop_corrupted_rows(df: pd.DataFrame) -> tuple:
    """
    Drop rows containing NaN or Inf values in any numeric column.

    These 26 rows are zero-duration flows where CICFlowMeter
    computed division-by-zero — a documented artefact (Engelen
    et al., 2021). Dropping is preferable to imputation because
    these are corrupted records, not randomly missing values.

    Returns
    -------
    df_clean     : pd.DataFrame  — rows without NaN or Inf
    n_dropped    : int           — number of rows dropped
    dropped_idx  : pd.Index      — original indices of dropped rows
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Identify rows with any NaN
    nan_mask = df[numeric_cols].isnull().any(axis=1)

    # Identify rows with any Inf
    inf_mask = np.isinf(
        df[numeric_cols].values
    ).any(axis=1)
    inf_mask = pd.Series(inf_mask, index=df.index)

    # Combined mask — any corrupted row
    bad_mask    = nan_mask | inf_mask
    dropped_idx = df.index[bad_mask]
    n_dropped   = int(bad_mask.sum())

    df_clean = df[~bad_mask].reset_index(drop=True)

    print(f"\n[drop_rows] Corrupted rows dropped: {n_dropped}")
    print(f"            Reason: NaN in IAT features + Inf in "
          f"rate features")
    print(f"            Root cause: zero-duration flows "
          f"(CICFlowMeter div-by-zero artefact)")
    print(f"            Rows remaining: {len(df_clean):,}")
    print(f"            Pct dropped: "
          f"{n_dropped / len(df) * 100:.3f}%")

    return df_clean, n_dropped, dropped_idx

# =============================================================
#  3. DROP NEAR-ZERO VARIANCE FEATURES
# =============================================================

def drop_low_variance(df: pd.DataFrame) -> tuple:
    """
    Drop numeric feature columns with standard deviation below
    VARIANCE_THRESHOLD (default 0.001).

    Low-variance features carry almost no discriminative signal
    and add noise to the correlation analysis.

    The label and timestamp columns are preserved throughout.

    Returns
    -------
    df_clean     : pd.DataFrame  — features with sufficient variance
    dropped_cols : list          — names of dropped columns
    variances    : pd.Series     — std of all numeric features
    """
    exclude = [LABEL_COLUMN]
    if TIMESTAMP_COL in df.columns:
        exclude.append(TIMESTAMP_COL)

    feature_cols = [c for c in df.select_dtypes(
        include=[np.number]).columns if c not in exclude]

    stds         = df[feature_cols].std()
    low_var_cols = stds[stds < VARIANCE_THRESHOLD].index.tolist()

    df_clean = df.drop(columns=low_var_cols)

    print(f"\n[low_var]   Near-zero variance features dropped: "
          f"{len(low_var_cols)}")
    if low_var_cols:
        for col in low_var_cols:
            print(f"            - {col}  (std={stds[col]:.6f})")
    print(f"            Features remaining: "
          f"{len(feature_cols) - len(low_var_cols)}")

    return df_clean, low_var_cols, stds

# =============================================================
#  4. DROP HIGHLY CORRELATED FEATURES
# =============================================================

def drop_high_correlation(df: pd.DataFrame) -> tuple:
    """
    Compute pairwise Pearson correlations among numeric features.
    For each pair with |correlation| > CORRELATION_THRESHOLD,
    drop the feature with lower variance (keeps more informative).

    This is the core feature selection step that replaces PCA —
    it reduces dimensionality while preserving interpretability.

    Returns
    -------
    df_clean     : pd.DataFrame  — decorrelated feature set
    dropped_cols : list          — names of dropped columns
    corr_matrix  : pd.DataFrame  — full correlation matrix
    """
    exclude = [LABEL_COLUMN]
    if TIMESTAMP_COL in df.columns:
        exclude.append(TIMESTAMP_COL)

    feature_cols = [c for c in df.select_dtypes(
        include=[np.number]).columns if c not in exclude]

    print(f"\n[corr]      Computing correlations for "
          f"{len(feature_cols)} features ...")

    corr_matrix = df[feature_cols].corr(method="pearson").abs()

    # Upper triangle mask — avoid duplicate pairs
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Variance of each feature (used for tie-breaking)
    variances   = df[feature_cols].var()

    dropped_cols = []
    to_drop_set  = set()

    # Iterate upper triangle — find correlated pairs
    for col in upper.columns:
        for row in upper.index:
            if col == row:
                continue
            val = upper.loc[row, col]
            if pd.isna(val):
                continue
            if float(val) > CORRELATION_THRESHOLD:
                # Drop the one with lower variance
                if variances[row] <= variances[col]:
                    to_drop_set.add(row)
                else:
                    to_drop_set.add(col)

    dropped_cols = list(to_drop_set)
    df_clean     = df.drop(columns=dropped_cols)

    remaining = len(feature_cols) - len(dropped_cols)
    print(f"[corr]      Threshold: |r| > {CORRELATION_THRESHOLD}")
    print(f"[corr]      Correlated features dropped: "
          f"{len(dropped_cols)}")
    print(f"[corr]      Features remaining: {remaining}")

    return df_clean, dropped_cols, corr_matrix

# =============================================================
#  5. CORRELATION HEATMAP PLOT
# =============================================================

def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                             feature_cols: list,
                             save: bool = True):
    """
    Plot correlation heatmap of retained features.
    Only shows features that survived correlation filtering.
    Saves to results/correlation_heatmap.png
    """
    # Subset to retained features only
    retained = [c for c in feature_cols
                if c in corr_matrix.columns]
    sub_corr = corr_matrix.loc[retained, retained]

    # Only plot if manageable size
    if len(retained) > 60:
        print(f"[plot]  Heatmap skipped — too many features "
              f"({len(retained)}) for readable plot.")
        print(f"        Saving numerical correlation summary instead.")
        out = os.path.join(RESULTS_DIR, "correlation_summary.csv")
        sub_corr.to_csv(out)
        print(f"[plot]  Correlation matrix saved → {out}")
        return

    fig, ax = plt.subplots(
        figsize=(max(10, len(retained) * 0.3),
                 max(8,  len(retained) * 0.3))
    )
    sns.heatmap(
        sub_corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        linecolor="white",
        annot=False,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(
        f"Feature correlation matrix — {len(retained)} retained features",
        fontsize=11, pad=10
    )
    ax.tick_params(axis="both", labelsize=6)
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "correlation_heatmap.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot]  Correlation heatmap saved → {out}")
    plt.show()
    plt.close()

# =============================================================
#  6. NORMALIZATION
# =============================================================

def normalize_features(df: pd.DataFrame) -> tuple:
    """
    Apply z-score normalization to all numeric feature columns.

    IMPORTANT: The scaler is fit here on the FULL cleaned dataset.
    In Stage 3, after train/val/test splitting, the scaler will be
    RE-FIT on training data only and re-applied to all splits to
    prevent data leakage. This call here is for inspection and
    saving the processed DataFrame only.

    Returns
    -------
    df_norm      : pd.DataFrame  — normalized feature DataFrame
    scaler       : StandardScaler — fitted on full cleaned data
    feature_cols : list           — final feature column names
    """
    exclude = [LABEL_COLUMN]
    if TIMESTAMP_COL in df.columns:
        exclude.append(TIMESTAMP_COL)

    feature_cols = [c for c in df.select_dtypes(
        include=[np.number]).columns if c not in exclude]

    scaler   = StandardScaler()
    df_norm  = df.copy()
    df_norm[feature_cols] = scaler.fit_transform(
        df[feature_cols].values
    )

    print(f"\n[norm]      Z-score normalization applied.")
    print(f"            Features normalized: {len(feature_cols)}")
    print(f"            Mean range: "
          f"[{df_norm[feature_cols].mean().min():.4f}, "
          f"{df_norm[feature_cols].mean().max():.4f}]")
    print(f"            Std range : "
          f"[{df_norm[feature_cols].std().min():.4f}, "
          f"{df_norm[feature_cols].std().max():.4f}]")

    return df_norm, scaler, feature_cols

# =============================================================
#  7. FEATURE SELECTION AUDIT REPORT
# =============================================================

def print_audit_report(original_cols: int,
                       after_str_drop: int,
                       after_row_drop: int,
                       after_var_drop: int,
                       after_corr_drop: int,
                       dropped_str: list,
                       dropped_var: list,
                       dropped_corr: list):
    """
    Print a structured audit trail of all feature selection steps.
    This maps directly to Section 4.2 of the paper.
    """
    sep = "=" * 62
    print(f"\n{sep}")
    print("  FEATURE SELECTION AUDIT REPORT")
    print(sep)
    print(f"\n  {'Step':<40} {'Features':>10}  {'Removed':>8}")
    print(f"  {'-'*40} {'-'*10}  {'-'*8}")
    print(f"  {'Original feature set':<40} "
          f"{original_cols:>10}  {'—':>8}")
    print(f"  {'After dropping string/object cols':<40} "
          f"{after_str_drop:>10}  "
          f"{original_cols - after_str_drop:>8}")
    print(f"  {'After dropping corrupted rows*':<40} "
          f"{'(rows)':>10}  "
          f"{after_row_drop:>8}")
    print(f"  {'After near-zero variance filter':<40} "
          f"{after_var_drop:>10}  "
          f"{after_str_drop - after_var_drop:>8}")
    print(f"  {'After correlation filter (|r|>0.95)':<40} "
          f"{after_corr_drop:>10}  "
          f"{after_var_drop - after_corr_drop:>8}")
    print(f"\n  * Row removal: 26 zero-duration flow artefacts")
    print(f"\n  Final feature set : {after_corr_drop} features")
    print(f"  Total features removed: "
          f"{original_cols - after_corr_drop}")
    print(f"\n{sep}\n")

# =============================================================
#  8. SAVE PREPROCESSED DATA
# =============================================================

def save_preprocessed(df_norm: pd.DataFrame,
                      feature_cols: list,
                      scaler: StandardScaler):
    """
    Save the preprocessed DataFrame, feature list, and scaler
    to DATA_DIR for use in Stage 3.
    """
    # Preprocessed CSV
    out_csv = os.path.join(DATA_DIR, "preprocessed.csv")
    df_norm.to_csv(out_csv, index=False)
    print(f"[save]  Preprocessed CSV → {out_csv}")

    # Feature names list
    out_feat = os.path.join(DATA_DIR, "feature_names.pkl")
    with open(out_feat, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"[save]  Feature names   → {out_feat}")

    # Scaler (will be re-fit in Stage 3 on train split only)
    out_scaler = os.path.join(DATA_DIR, "scaler_full.pkl")
    with open(out_scaler, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[save]  Scaler (full)   → {out_scaler}")

# =============================================================
#  9. MAIN PIPELINE
# =============================================================

def run_stage2(df: pd.DataFrame) -> tuple:
    """
    Execute complete Stage 2 preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged DataFrame from stage1_setup.run_stage1()

    Returns
    -------
    df_norm      : pd.DataFrame   — cleaned, normalized DataFrame
    feature_cols : list           — final feature column names
    scaler       : StandardScaler — fitted scaler
    summary      : dict           — preprocessing statistics
    """
    print("\n" + "=" * 62)
    print("  STAGE 2 — PREPROCESSING AND FEATURE SELECTION")
    print("=" * 62 + "\n")

    original_n_cols = df.shape[1] - 1  # exclude label

    # ── Step 1: Drop non-numeric columns ──────────────────────
    df, dropped_str = drop_non_numeric(df)
    after_str_drop  = (df.shape[1] - 1  # exclude label
                       - (1 if TIMESTAMP_COL in df.columns else 0))

    # ── Step 2: Drop corrupted rows ───────────────────────────
    df, n_dropped_rows, _ = drop_corrupted_rows(df)
    after_row_drop = n_dropped_rows

    # ── Step 3: Drop near-zero variance features ──────────────
    df, dropped_var, stds = drop_low_variance(df)
    after_var_drop = (df.shape[1] - 1
                      - (1 if TIMESTAMP_COL in df.columns else 0))

    # ── Step 4: Drop highly correlated features ───────────────
    df, dropped_corr, corr_matrix = drop_high_correlation(df)
    after_corr_drop = (df.shape[1] - 1
                       - (1 if TIMESTAMP_COL in df.columns else 0))

    # ── Step 5: Plot correlation heatmap ──────────────────────
    exclude = [LABEL_COLUMN]
    if TIMESTAMP_COL in df.columns:
        exclude.append(TIMESTAMP_COL)
    retained_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    plot_correlation_heatmap(
        corr_matrix, retained_features, save=True
    )

    # ── Step 6: Normalize ─────────────────────────────────────
    df_norm, scaler, feature_cols = normalize_features(df)

    # ── Step 7: Audit report ──────────────────────────────────
    print_audit_report(
        original_cols   = original_n_cols,
        after_str_drop  = after_str_drop,
        after_row_drop  = after_row_drop,
        after_var_drop  = after_var_drop,
        after_corr_drop = after_corr_drop,
        dropped_str     = dropped_str,
        dropped_var     = dropped_var,
        dropped_corr    = dropped_corr,
    )

    # ── Step 8: Save ──────────────────────────────────────────
    save_preprocessed(df_norm, feature_cols, scaler)

    # ── Summary dict ──────────────────────────────────────────
    summary = {
        "original_features"   : original_n_cols,
        "final_features"      : len(feature_cols),
        "dropped_string"      : len(dropped_str),
        "dropped_low_var"     : len(dropped_var),
        "dropped_correlated"  : len(dropped_corr),
        "dropped_rows"        : after_row_drop,
        "final_rows"          : len(df_norm),
        "feature_names"       : feature_cols,
    }

    print(f"\n[stage2] ✓ Complete.")
    print(f"         Input shape   : {df.shape}")
    print(f"         Output shape  : {df_norm.shape}")
    print(f"         Features kept : {len(feature_cols)}")
    print(f"         → Pass df_norm to "
          f"stage3_sequences.run_stage3(df_norm, feature_cols)\n")

    return df_norm, feature_cols, scaler, summary


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  From terminal (venv activated):
#    cd ~/Documents/kubernetes_anomaly_detection
#    python scripts/stage2_preprocessing.py
#
#  From notebook or after stage1:
#    from scripts.stage2_preprocessing import run_stage2
#    df_norm, feature_cols, scaler, summary2 = run_stage2(df)
#
# =============================================================

if __name__ == "__main__":
    # Load merged raw CSV saved by Stage 1
    import pickle
    raw_path = os.path.join(DATA_DIR, "merged_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"merged_raw.csv not found at {raw_path}\n"
            f"Run stage1_setup.py first."
        )
    print(f"[load]  Reading merged_raw.csv ...")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"        Shape: {df.shape}")

    df_norm, feature_cols, scaler, summary = run_stage2(df)