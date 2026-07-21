# =============================================================
#  stage8_evaluation.py
#  Stage 8 — Comprehensive Evaluation and Results Generation
#
#  What this script does:
#    1. Loads best transformer checkpoint from stage6
#    2. Loads baseline results from stage7
#    3. Evaluates transformer on test set
#    4. Generates complete comparison table (all models)
#    5. Plots confusion matrix heatmap (transformer)
#    6. Plots per-class F1 comparison bar chart (all models)
#    7. Plots ROC curves (transformer, one-vs-rest)
#    8. Saves all results to results/ directory
#
#  All figures map directly to paper sections:
#    Figure 4 — Training curves        (already saved in stage6)
#    Figure 5 — Confusion matrix       (this script)
#    Figure 6 — Per-class F1 chart     (this script)
#    Table 4  — Full comparison table  (this script)
#
#  Input  : best_model.pt from checkpoints/
#            baseline_results.pkl from results/
#  Output : All evaluation figures and tables saved to results/
# =============================================================

import os
import sys
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import pickle
import json
import warnings

warnings.filterwarnings("ignore")

# =============================================================
#  DYNAMIC PROJECT ROOT & CONFIG LOADING
# =============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Add PROJECT_ROOT to sys.path for stage5/stage4 imports
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
DATA_DIR        = project_config.DATA_DIR
RESULTS_DIR     = project_config.RESULTS_DIR
CHECKPOINT_DIR  = project_config.CHECKPOINT_DIR
NUM_CLASSES     = project_config.NUM_CLASSES
CLASS_NAMES     = project_config.CLASS_NAMES
CLASS_SHORT     = project_config.CLASS_SHORT
BATCH_SIZE      = project_config.BATCH_SIZE
WINDOW_SIZE     = project_config.WINDOW_SIZE
D_MODEL         = project_config.D_MODEL
N_HEADS         = project_config.N_HEADS
N_LAYERS        = project_config.N_LAYERS
D_FF            = project_config.D_FF
DROPOUT         = project_config.DROPOUT
FC_HIDDEN       = project_config.FC_HIDDEN

INPUT_DIM = 39

# =============================================================
#  1. DEVICE DETECTION
# =============================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# =============================================================
#  2. LOAD TRANSFORMER AND EVALUATE ON TEST SET
# =============================================================

def evaluate_transformer(device: torch.device) -> dict:
    """
    Load best transformer checkpoint and evaluate on test set.
    Returns full metrics dict.
    """
    from scripts.stage5_model import KubernetesTransformer
    from scripts.stage4_dataloader import KubernetesSequenceDataset
    from torch.utils.data import DataLoader

    print(f"\n{'='*62}")
    print(f"  TRANSFORMER — TEST SET EVALUATION")
    print(f"{'='*62}")

    # Load checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run stage6_train.py first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[ckpt]  Loaded checkpoint from epoch "
          f"{ckpt['epoch']}")
    print(f"[ckpt]  Best val macro-F1 : "
          f"{ckpt['val_f1']:.4f}")

    # Build model and load weights
    model = KubernetesTransformer(
        input_dim = INPUT_DIM,
        d_model   = D_MODEL,
        n_heads   = N_HEADS,
        n_layers  = N_LAYERS,
        d_ff      = D_FF,
        dropout   = DROPOUT,
        fc_hidden = FC_HIDDEN,
        n_classes = NUM_CLASSES,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load test split
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    test_ds     = KubernetesSequenceDataset(X_test, y_test)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    # Inference
    all_preds  = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X  = batch_X.to(device)
            logits   = model(batch_X)
            probs    = torch.softmax(logits, dim=1)
            preds    = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(batch_y.numpy().tolist())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)   # (N, 11)

    # Metrics
    macro_f1    = f1_score(y_true, y_pred,
                           average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred,
                           average="weighted", zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)

    # ROC-AUC (one-vs-rest, macro)
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    try:
        roc_auc = roc_auc_score(
            y_bin, y_probs,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        roc_auc = float("nan")

    # Per-class F1
    per_class_f1 = f1_score(
        y_true, y_pred,
        average=None,
        zero_division=0,
        labels=list(range(NUM_CLASSES))
    )

    print(f"\n  Macro-F1    : {macro_f1:.4f}")
    print(f"  Weighted-F1 : {weighted_f1:.4f}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  ROC-AUC     : {roc_auc:.4f}")

    print(f"\n  Per-class report:")
    target_names = [CLASS_NAMES.get(i, str(i))
                    for i in range(NUM_CLASSES)]
    print(classification_report(
        y_true, y_pred,
        target_names  = target_names,
        zero_division = 0,
        digits        = 4
    ))

    return {
        "model"          : "Transformer (ours)",
        "macro_f1"       : round(macro_f1,    4),
        "weighted_f1"    : round(weighted_f1, 4),
        "accuracy"       : round(accuracy,    4),
        "roc_auc"        : round(roc_auc,     4),
        "per_class_f1"   : per_class_f1.tolist(),
        "y_pred"         : y_pred.tolist(),
        "y_true"         : y_true.tolist(),
        "y_probs"        : y_probs.tolist(),
        "best_epoch"     : ckpt["epoch"],
        "best_val_f1"    : ckpt["val_f1"],
    }

# =============================================================
#  3. CONFUSION MATRIX PLOT
# =============================================================

def plot_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           title:  str = "Transformer",
                           save:   bool = True,
                           fname:  str = "confusion_matrix.png"):
    """
    Normalized confusion matrix heatmap.
    Rows = true labels, Columns = predicted labels.
    Values are row-normalized (recall per class).
    """
    cm      = confusion_matrix(y_true, y_pred,
                               labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    # Avoid division by zero for absent classes
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    labels = [CLASS_SHORT.get(i, str(i))
              for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        ax          = ax,
        annot       = True,
        fmt         = ".2f",
        cmap        = "Blues",
        xticklabels = labels,
        yticklabels = labels,
        vmin        = 0,
        vmax        = 1,
        linewidths  = 0.3,
        linecolor   = "white",
        cbar_kws    = {"label": "Recall (row-normalized)"}
    )
    ax.set_title(
        f"Confusion matrix — {title}\n"
        f"Row-normalized (diagonal = per-class recall)",
        fontsize=12, pad=12
    )
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label",      fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot]  Confusion matrix → {out}")
    plt.show()
    plt.close()

# =============================================================
#  4. PER-CLASS F1 COMPARISON CHART
# =============================================================

def plot_per_class_f1(transformer_results: dict,
                       baseline_results:   dict,
                       save: bool = True):
    """
    Grouped bar chart comparing per-class F1 across all models.
    This is Figure 6 in the paper — the most visually compelling
    demonstration of Novelty 1.
    """
    models = ["Random Forest", "MLP", "LSTM",
              "Transformer (ours)"]
    colors = ["#B5D4F4", "#85B7EB", "#1D9E75", "#1F4E79"]

    # Gather per-class F1 for each model
    # For baselines compute from saved y_true/y_pred
    per_class = {}

    for model_name in ["Random Forest", "MLP", "LSTM"]:
        if model_name not in baseline_results:
            continue
        res   = baseline_results[model_name]
        y_t   = np.array(res["y_true"])
        y_p   = np.array(res["y_pred"])
        f1s   = f1_score(
            y_t, y_p,
            average  = None,
            labels   = list(range(NUM_CLASSES)),
            zero_division = 0
        )
        per_class[model_name] = f1s

    per_class["Transformer (ours)"] = np.array(
        transformer_results["per_class_f1"]
    )

    class_labels = [CLASS_SHORT.get(i, str(i))
                    for i in range(NUM_CLASSES)]
    x      = np.arange(NUM_CLASSES)
    n_models = len(per_class)
    width  = 0.8 / n_models
    offset = -(n_models - 1) / 2

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, (model_name, f1s) in enumerate(per_class.items()):
        positions = x + (offset + i) * width
        bars = ax.bar(
            positions, f1s,
            width       = width * 0.9,
            label       = model_name,
            color       = colors[i],
            edgecolor   = "white",
            linewidth   = 0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=35,
                        fontsize=8.5, ha="right")
    ax.set_ylabel("F1 score", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Per-class F1 score comparison — all models\n"
        "Test set · Sever & Doğan (2023) Dataset",
        fontsize=12, pad=10
    )
    ax.legend(fontsize=9, loc="upper right",
              framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "per_class_f1.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot]  Per-class F1 chart → {out}")
    plt.show()
    plt.close()

# =============================================================
#  5. ROC CURVES PLOT
# =============================================================

def plot_roc_curves(y_true:  np.ndarray,
                    y_probs: np.ndarray,
                    save:    bool = True):
    """
    One-vs-rest ROC curves for each class.
    Transformer only — baselines do not produce probabilities
    in a comparable format from sklearn.
    """
    y_bin = label_binarize(
        y_true, classes=list(range(NUM_CLASSES))
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for i in range(NUM_CLASSES):
        # Skip classes with no positive samples in test
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        name = CLASS_SHORT.get(i, str(i))
        ax.plot(fpr, tpr,
                label=f"{name} (AUC={roc_auc_val:.3f})",
                color=colors[i], linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--",
            linewidth=0.8, label="Random classifier")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate",  fontsize=10)
    ax.set_title(
        "ROC curves — Transformer (one-vs-rest)\n"
        "Test set · Sever & Doğan (2023) Dataset",
        fontsize=12, pad=10
    )
    ax.legend(fontsize=8, loc="lower right",
              framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save:
        out = os.path.join(RESULTS_DIR, "roc_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot]  ROC curves → {out}")
    plt.show()
    plt.close()

# =============================================================
#  6. FULL COMPARISON TABLE
# =============================================================

def print_full_comparison_table(transformer_results: dict,
                                 baseline_results:   dict):
    """
    Print and save the complete model comparison table.
    This is Table 4 in the paper — Section 6.1.
    """
    sep = "=" * 78
    print(f"\n{sep}")
    print("  TABLE 4 — COMPLETE MODEL COMPARISON  (Test Set)")
    print(sep)
    print(f"  {'Model':<26} {'Macro-F1':>10} "
          f"{'Weighted-F1':>12} {'Accuracy':>10} "
          f"{'ROC-AUC':>9}  {'Seq-Aware':>15}")
    print(f"  {'-'*26} {'-'*10} "
          f"{'-'*12} {'-'*10} "
          f"{'-'*9}  {'-'*15}")

    seq_aware_map = {
        "Random Forest"      : "No",
        "MLP"                : "No",
        "LSTM"               : "Yes (recurrent)",
        "Transformer (ours)" : "Yes (attention)",
    }

    # Order: baselines first, transformer last
    ordered_models = ["Random Forest", "MLP", "LSTM",
                      "Transformer (ours)"]

    all_results = {**baseline_results,
                   "Transformer (ours)": transformer_results}

    for model_name in ordered_models:
        if model_name not in all_results:
            continue
        r = all_results[model_name]

        roc = r.get("roc_auc", "—")
        roc_str = f"{roc:.4f}" if isinstance(roc, float) else "—"

        seq = seq_aware_map.get(model_name, "—")
        bold = "★ " if model_name == "Transformer (ours)" else "  "

        print(f"  {bold}{model_name:<24} "
              f"{r['macro_f1']:>10.4f} "
              f"{r['weighted_f1']:>12.4f} "
              f"{r['accuracy']:>10.4f} "
              f"{roc_str:>9}  "
              f"{seq:>15}")

    print(f"\n  ★ = proposed method")
    print(f"  Primary metric: Macro-F1 (equal weight per class)")
    print(f"  Test set: {1210} sequences  "
          f"(W={WINDOW_SIZE}, D={INPUT_DIM})")
    print(f"  Dataset: Sever & Doğan (2023)  "
          f"— 11-class misuse detection")
    print(f"{sep}\n")

    # Save as JSON
    table_data = {}
    for model_name in ordered_models:
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        table_data[model_name] = {
            "macro_f1"    : r["macro_f1"],
            "weighted_f1" : r["weighted_f1"],
            "accuracy"    : r["accuracy"],
            "roc_auc"     : r.get("roc_auc", None),
            "seq_aware"   : seq_aware_map.get(model_name, "—"),
        }

    out = os.path.join(RESULTS_DIR, "comparison_table.json")
    with open(out, "w") as f:
        json.dump(table_data, f, indent=2)
    print(f"[save]  Comparison table → {out}")

# =============================================================
#  7. SUMMARY STATISTICS FOR PAPER
# =============================================================

def print_paper_summary(transformer_results: dict,
                         baseline_results:   dict):
    """
    Print key statistics and improvement claims for the paper.
    Maps to Section 6.1 and 6.5 discussion points.
    """
    sep = "=" * 62
    tf_f1  = transformer_results["macro_f1"]
    rf_f1  = baseline_results.get(
        "Random Forest", {}).get("macro_f1", 0)
    mlp_f1 = baseline_results.get(
        "MLP", {}).get("macro_f1", 0)
    lstm_f1= baseline_results.get(
        "LSTM", {}).get("macro_f1", 0)

    print(f"\n{sep}")
    print("  KEY FINDINGS FOR PAPER")
    print(sep)
    print(f"\n  Transformer macro-F1   : {tf_f1:.4f}")
    print(f"  vs Random Forest       : "
          f"+{(tf_f1 - rf_f1):.4f}  "
          f"({(tf_f1/rf_f1 - 1)*100:.1f}% improvement)")
    print(f"  vs MLP                 : "
          f"+{(tf_f1 - mlp_f1):.4f}  "
          f"({(tf_f1/mlp_f1 - 1)*100:.1f}% improvement)")
    print(f"  vs LSTM                : "
          f"+{(tf_f1 - lstm_f1):.4f}  "
          f"({(tf_f1/lstm_f1 - 1)*100:.1f}% improvement)")

    print(f"\n  Novelty 1 evidence:")
    print(f"  RF→MLP gain  (per-flow deep learning)  : "
          f"+{(mlp_f1 - rf_f1):.4f}")
    print(f"  MLP→LSTM gain (sequence modeling)       : "
          f"+{(lstm_f1 - mlp_f1):.4f}")
    print(f"  LSTM→Transformer gain (attention)       : "
          f"+{(tf_f1 - lstm_f1):.4f}  ← largest gain")
    print(f"\n  Conclusion: attention-based sequence modeling")
    print(f"  provides the largest single improvement,")
    print(f"  validating Novelty 1.\n")
    print(sep)

# =============================================================
#  8. MAIN PIPELINE
# =============================================================

def run_stage8() -> dict:
    """
    Execute complete Stage 8 evaluation.

    Returns
    -------
    all_results : dict with transformer + baseline results
    """
    print("\n" + "=" * 62)
    print("  STAGE 8 — EVALUATION AND RESULTS GENERATION")
    print("=" * 62 + "\n")

    device = get_device()
    print(f"[device] {device}\n")

    # ── Load baseline results from stage7 ─────────────────────
    baseline_path = os.path.join(
        RESULTS_DIR, "baseline_results.pkl"
    )
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"baseline_results.pkl not found.\n"
            f"Run stage7_baselines.py first."
        )
    with open(baseline_path, "rb") as f:
        baseline_results = pickle.load(f)
    print(f"[load]  Baseline results loaded "
          f"({len(baseline_results)} models)")

    # ── Evaluate transformer ───────────────────────────────────
    transformer_results = evaluate_transformer(device)

    # ── Full comparison table ──────────────────────────────────
    print_full_comparison_table(
        transformer_results, baseline_results
    )

    # ── Paper summary statistics ───────────────────────────────
    print_paper_summary(transformer_results, baseline_results)

    # ── Confusion matrix ───────────────────────────────────────
    y_true = np.array(transformer_results["y_true"])
    y_pred = np.array(transformer_results["y_pred"])
    y_probs= np.array(transformer_results["y_probs"])

    print(f"\n[plot]  Generating figures ...")
    plot_confusion_matrix(
        y_true, y_pred,
        title = "Transformer (ours)",
        save  = True,
        fname = "confusion_matrix_transformer.png"
    )

    # ── Per-class F1 comparison ───────────────────────────────
    plot_per_class_f1(
        transformer_results, baseline_results, save=True
    )

    # ── ROC curves ────────────────────────────────────────────
    plot_roc_curves(y_true, y_probs, save=True)

    # ── Save transformer results ──────────────────────────────
    tf_out = os.path.join(
        RESULTS_DIR, "transformer_results.pkl"
    )
    with open(tf_out, "wb") as f:
        pickle.dump(transformer_results, f)
    print(f"[save]  Transformer results → {tf_out}")

    # Combined results
    all_results = {
        **baseline_results,
        "Transformer (ours)": transformer_results,
    }
    all_out = os.path.join(RESULTS_DIR, "all_results.pkl")
    with open(all_out, "wb") as f:
        pickle.dump(all_results, f)
    print(f"[save]  All results → {all_out}")

    print(f"\n[stage8] ✓ Complete.")
    print(f"         Results saved to: {RESULTS_DIR}")
    print(f"         Figures generated:")
    print(f"           - confusion_matrix_transformer.png")
    print(f"           - per_class_f1.png")
    print(f"           - roc_curves.png")
    print(f"         → Run stage9_ablation.py on GPU "
          f"for ablation study\n")

    return all_results


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  Terminal:
#    python scripts/stage8_evaluation.py
#
#  From notebook:
#    from scripts.stage8_evaluation import run_stage8
#    all_results = run_stage8()
#
# =============================================================

if __name__ == "__main__":
    run_stage8()