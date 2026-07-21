# =============================================================
#  stage5_model.py
#  Stage 5 — Transformer Encoder Model Architecture
#
#  Implements the proposed sequence-aware transformer for
#  Kubernetes misuse detection (Novelty 1).
#
#  Architecture (maps to paper Section 4.4 and 4.5):
#
#    Input: (B, W, D) = (batch, 16, 39)
#      │
#      ├── Linear projection      D → d_model       (4.4.1)
#      ├── Sinusoidal pos. enc.   d_model            (4.4.2)
#      │
#      └── × N encoder layers                        (4.5)
#            ├── Multi-head self-attention  (h heads)
#            ├── Add & Layer Norm
#            ├── Feed-forward  (d_ff, ReLU)
#            └── Add & Layer Norm
#      │
#      ├── Global average pooling  (W, d_model) → d_model
#      ├── FC layer + ReLU + Dropout
#      └── Output layer            d_model → 11 classes
#
#  Design decisions:
#    - Pure encoder (no decoder) — classification only
#    - Sinusoidal positional encoding — no learned parameters
#    - Global average pooling — treats all positions equally
#    - No PCA/hybrid LSTM — differentiates from Allahabadi [34]
#
#  Input  : config.py hyperparameters
#  Output : KubernetesTransformer model ready for training
# =============================================================

import os
import sys
import math
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

D_MODEL = project_config.D_MODEL
N_HEADS = project_config.N_HEADS
N_LAYERS = project_config.N_LAYERS
D_FF = project_config.D_FF
DROPOUT = project_config.DROPOUT
FC_HIDDEN = project_config.FC_HIDDEN
NUM_CLASSES = project_config.NUM_CLASSES
WINDOW_SIZE = project_config.WINDOW_SIZE
BATCH_SIZE = project_config.BATCH_SIZE

# Confirmed after Stage 2
INPUT_DIM = 39    # D — features after preprocessing

# =============================================================
#  1. SINUSOIDAL POSITIONAL ENCODING
# =============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    Adds position-dependent signal to each flow embedding so
    the transformer can distinguish flow position within the
    sequence. Required because self-attention is permutation-
    invariant without positional information.

    Formula:
      PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    This is deterministic (no learned parameters), generalizes
    to unseen sequence lengths, and encodes relative positions
    as linear transformations — suitable for attack progression
    sequences where position ordering carries temporal meaning.

    Parameters
    ----------
    d_model  : int   embedding dimension
    max_len  : int   maximum sequence length supported
    dropout  : float dropout applied after adding encoding
    """

    def __init__(self,
                 d_model: int,
                 max_len: int = 512,
                 dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build encoding matrix (max_len, d_model)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        # Register as buffer — saved with model but not a parameter
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, d_model)

        Returns
        -------
        x : (B, W, d_model)  with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# =============================================================
#  2. TRANSFORMER ENCODER LAYER
# =============================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Sublayer 1: Multi-head self-attention + residual + LayerNorm
    Sublayer 2: Position-wise FFN       + residual + LayerNorm

    Pre-norm variant (LayerNorm before sublayer) is used for
    more stable training with small datasets.

    Parameters
    ----------
    d_model  : int    model dimension
    n_heads  : int    number of attention heads
    d_ff     : int    feed-forward inner dimension
    dropout  : float  dropout rate
    """

    def __init__(self,
                 d_model: int = D_MODEL,
                 n_heads: int = N_HEADS,
                 d_ff:    int = D_FF,
                 dropout: float = DROPOUT):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim    = d_model,
            num_heads    = n_heads,
            dropout      = dropout,
            batch_first  = True,   # input shape (B, W, d_model)
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalizations (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, d_model)

        Returns
        -------
        x : (B, W, d_model)
        """
        # ── Sublayer 1: Multi-head self-attention ─────────────
        # Pre-norm: normalize before attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            query = x_norm,
            key   = x_norm,
            value = x_norm,
        )
        x = x + self.dropout(attn_out)   # residual connection

        # ── Sublayer 2: Feed-forward network ──────────────────
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out                   # residual connection

        return x

# =============================================================
#  3. FULL TRANSFORMER MODEL
# =============================================================

class KubernetesTransformer(nn.Module):
    """
    Sequence-aware transformer encoder for Kubernetes
    misuse detection.

    Full architecture:
      1. Linear projection      : D=39 → d_model=128
      2. Positional encoding    : sinusoidal, no parameters
      3. Encoder stack          : N=4 layers
      4. Global average pooling : (W, d_model) → d_model
      5. Classification head    : d_model → FC → 11 classes

    Parameters
    ----------
    input_dim  : int   D — number of input features (39)
    d_model    : int   embedding dimension (128)
    n_heads    : int   attention heads (8)
    n_layers   : int   encoder layers (4)
    d_ff       : int   feed-forward inner dim (512)
    dropout    : float dropout rate (0.1)
    fc_hidden  : int   FC layer size (128)
    n_classes  : int   output classes (11)
    max_len    : int   max sequence length supported
    """

    def __init__(self,
                 input_dim: int = INPUT_DIM,
                 d_model:   int = D_MODEL,
                 n_heads:   int = N_HEADS,
                 n_layers:  int = N_LAYERS,
                 d_ff:      int = D_FF,
                 dropout:   float = DROPOUT,
                 fc_hidden: int = FC_HIDDEN,
                 n_classes: int = NUM_CLASSES,
                 max_len:   int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.n_layers  = n_layers
        self.n_classes = n_classes

        # ── 1. Linear input projection ────────────────────────
        # Maps D-dim flow feature vector to d_model-dim embedding
        # This is the entry point for continuous numerical features
        # (analogous to patch embedding in ViT)
        self.input_projection = nn.Linear(input_dim, d_model)

        # ── 2. Positional encoding ────────────────────────────
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model = d_model,
            max_len = max_len,
            dropout = dropout,
        )

        # ── 3. Transformer encoder stack ─────────────────────
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model = d_model,
                n_heads = n_heads,
                d_ff    = d_ff,
                dropout = dropout,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm after encoder stack
        self.encoder_norm = nn.LayerNorm(d_model)

        # ── 4. Classification head ────────────────────────────
        # Global average pooling is applied in forward()
        # FC layer + dropout + output layer
        self.classifier = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

        # ── Weight initialization ─────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for stable training.
        Linear layers: Xavier uniform initialization.
        LayerNorm: standard (ones/zeros).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, W, D)  — batch of flow sequences

        Returns
        -------
        logits : (B, n_classes)  — raw class scores
                 Apply softmax for probabilities.
                 Use argmax for predicted class.
        """
        # ── Step 1: Linear projection ─────────────────────────
        # (B, W, D) → (B, W, d_model)
        x = self.input_projection(x)

        # ── Step 2: Positional encoding ───────────────────────
        # (B, W, d_model) → (B, W, d_model)
        x = self.pos_encoding(x)

        # ── Step 3: Encoder stack ─────────────────────────────
        # Each layer: (B, W, d_model) → (B, W, d_model)
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.encoder_norm(x)

        # ── Step 4: Global average pooling ────────────────────
        # (B, W, d_model) → (B, d_model)
        # Averages across the sequence dimension W
        x = x.mean(dim=1)

        # ── Step 5: Classification head ───────────────────────
        # (B, d_model) → (B, n_classes)
        logits = self.classifier(x)

        return logits

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )

    def get_attention_weights(self,
                               x: torch.Tensor) -> list:
        """
        Extract attention weights from all encoder layers.
        Used for interpretability analysis in Section 6.

        Parameters
        ----------
        x : (B, W, D)

        Returns
        -------
        attn_weights : list of (B, n_heads, W, W) tensors
                       one per encoder layer
        """
        attn_weights = []

        x = self.input_projection(x)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x_norm = layer.norm1(x)
            _, weights = layer.self_attn(
                query             = x_norm,
                key               = x_norm,
                value             = x_norm,
                need_weights      = True,
                average_attn_weights = False,  # keep per-head
            )
            attn_weights.append(weights.detach().cpu())

            # Continue forward pass
            attn_out, _ = layer.self_attn(
                query = x_norm,
                key   = x_norm,
                value = x_norm,
            )
            x = x + layer.dropout(attn_out)
            x_norm = layer.norm2(x)
            x = x + layer.ff(x_norm)

        return attn_weights

# =============================================================
#  4. MODEL SUMMARY
# =============================================================

def print_model_summary(model: KubernetesTransformer):
    """Print a structured model summary for the paper."""
    sep = "=" * 62
    print(f"\n{sep}")
    print("  MODEL ARCHITECTURE SUMMARY")
    print(sep)
    print(f"\n  KubernetesTransformer")
    print(f"  {'Component':<35} {'Shape / Config'}")
    print(f"  {'-'*35} {'-'*25}")
    print(f"  {'Input':<35} (B, W={WINDOW_SIZE}, D={model.input_dim})")
    print(f"  {'Linear projection':<35} "
          f"{model.input_dim} → {model.d_model}")
    print(f"  {'Positional encoding':<35} sinusoidal, no params")
    print(f"  {'Encoder layers':<35} × {model.n_layers}")
    print(f"  {'  Multi-head attention':<35} "
          f"h={model.n_heads}, d_k={model.d_model//model.n_heads}")
    print(f"  {'  Feed-forward':<35} "
          f"{model.d_model} → {D_FF} → {model.d_model}")
    print(f"  {'  Dropout':<35} {DROPOUT}")
    print(f"  {'Global average pooling':<35} "
          f"(B, {WINDOW_SIZE}, {model.d_model}) → "
          f"(B, {model.d_model})")
    print(f"  {'FC layer':<35} "
          f"{model.d_model} → {FC_HIDDEN}")
    print(f"  {'Output layer':<35} "
          f"{FC_HIDDEN} → {model.n_classes} classes")
    print(f"\n  Total trainable parameters: "
          f"{model.count_parameters():,}")
    print(f"{sep}\n")

# =============================================================
#  5. MODEL INSTANTIATION AND VERIFICATION
# =============================================================

def build_model(device: torch.device = None) -> tuple:
    """
    Build and verify the transformer model.

    Runs a forward pass with a dummy batch to confirm
    all shapes are correct before training.

    Returns
    -------
    model  : KubernetesTransformer on target device
    device : torch.device
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

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

    print_model_summary(model)

    # ── Forward pass verification ─────────────────────────────
    print(f"[verify] Running forward pass with dummy batch ...")
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(
            BATCH_SIZE, WINDOW_SIZE, INPUT_DIM
        ).to(device)
        logits = model(dummy)

    print(f"         Input  shape : {dummy.shape}")
    print(f"         Output shape : {logits.shape}")
    print(f"         Expected     : "
          f"({BATCH_SIZE}, {NUM_CLASSES})")

    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), \
        f"Output shape mismatch: {logits.shape}"

    print(f"[verify] ✓ Forward pass successful.")
    print(f"         Model ready for training on {device}\n")

    return model, device


# =============================================================
#  HOW TO RUN
# =============================================================
#
#  Terminal:
#    python scripts/stage5_model.py
#
#  In training script (stage6):
#    from scripts.stage5_model import build_model
#    model, device = build_model(device)
#
# =============================================================

if __name__ == "__main__":
    model, device = build_model()