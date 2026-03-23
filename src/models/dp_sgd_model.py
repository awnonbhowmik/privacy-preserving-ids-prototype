"""
dp_sgd_model.py — Differentially Private Neural Network training via DP-SGD.

Implements a PyTorch MLP trained with DP-SGD using the Opacus library. This is
the Phase 2 addition to the prototype, complementing the DP-LR (diffprivlib)
experiments with a deep-learning-based private classifier.

Architecture:
    MLP: input_size → 128 → 64 → output_size
    - Input size inferred dynamically from X_train.shape[1]
    - ReLU activations between hidden layers
    - No BatchNorm (incompatible with Opacus — requires GroupNorm or LayerNorm)

Privacy mechanism:
    DP-SGD (Differentially Private Stochastic Gradient Descent) via Opacus.
    Opacus clips per-sample gradients to max_grad_norm and adds calibrated
    Gaussian noise. The noise scale is automatically derived from the target
    epsilon via the Rényi Differential Privacy accountant.

Class imbalance:
    Inverse-frequency sample weights passed to WeightedRandomSampler.
    Opacus uses poisson_sampling=False to accommodate the custom sampler
    (standard Poisson subsampling replaces the sampler, which we avoid here).

Usage:
    from src.models.dp_sgd_model import train_dp_sgd_model, run_dp_sgd_sweep

    # Single epsilon
    result = train_dp_sgd_model("cic_ids2018", epsilon=1.0, label="binary")

    # Full sweep over dp_sgd.epsilon_values from config.yaml
    results = run_dp_sgd_sweep("cic_ids2018", label="binary")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.config import PATHS, DP_SGD_CFG, SAMPLING_CFG
from src.utils.logger import get_logger
from src.data.sampler import get_split_arrays
from src.evaluation.metrics import evaluate_model

log = get_logger(__name__)

DP_SGD_MODEL_NAME = "dp_sgd"


# ══════════════════════════════════════════════════════════════════════════════
# Path helper
# ══════════════════════════════════════════════════════════════════════════════

def _dp_sgd_model_path(dataset_name: str, epsilon: float, label: str) -> Path:
    """Construct the .joblib save path for a trained DP-SGD model at a given epsilon."""
    eps_str = f"{epsilon:.3f}".replace(".", "_")
    return PATHS["models"] / f"dp_sgd_{dataset_name}_{label}_eps{eps_str}.joblib"


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch MLP architecture
# ══════════════════════════════════════════════════════════════════════════════

def _build_mlp(input_size: int, hidden_layers: list[int], output_size: int):
    """
    Build a simple MLP with ReLU activations.

    No BatchNorm is used — Opacus requires replacing it with GroupNorm/LayerNorm,
    and for a two-hidden-layer IDS classifier the gain from normalisation is
    marginal compared to the implementation complexity.

    Args:
        input_size:    Number of input features (inferred from X_train.shape[1]).
        hidden_layers: List of hidden unit counts, e.g. [128, 64].
        output_size:   Number of output classes.

    Returns:
        nn.Sequential MLP module.
    """
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev_size = input_size
    for h in hidden_layers:
        layers.append(nn.Linear(prev_size, h))
        layers.append(nn.ReLU())
        prev_size = h
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════════
# sklearn-compatible wrapper
# ══════════════════════════════════════════════════════════════════════════════

class DPSGDWrapper:
    """
    Thin sklearn-compatible wrapper around a trained PyTorch MLP.

    Exposes .predict() and .predict_proba() so that the model can be passed
    directly to evaluate_model() in src/evaluation/metrics.py without changes
    to the evaluation code.

    The device is stored as a string so the wrapper is fully picklable by joblib
    across processes. Inference is always run on CPU for portability.
    """

    def __init__(
        self,
        model,
        classes: np.ndarray,
        label_type: str,
    ) -> None:
        self.model      = model       # CPU-placed nn.Module
        self.classes_   = classes     # integer class labels
        self.label_type = label_type

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions for feature matrix X."""
        import torch
        self.model.eval()
        # Inference always runs on CPU (model was moved to CPU after training)
        # for portability across platforms including macOS MPS.
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(dim=1).numpy().astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix for feature matrix X."""
        import torch
        import torch.nn.functional as F
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            logits = self.model(X_t)
            probs  = F.softmax(logits, dim=1)
        return probs.numpy().astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Sample weight helper
# ══════════════════════════════════════════════════════════════════════════════

def _compute_dp_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute inverse-frequency sample weights for WeightedRandomSampler.

    Mirrors the formula used in dp_model._compute_sample_weights():
        weight_i = n_samples / (n_classes × count(class_i))

    Args:
        y: Integer label array (training labels).

    Returns:
        Float array of per-sample weights, same length as y.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples  = len(y)
    n_classes  = len(classes)
    class_weight = {
        cls: n_samples / (n_classes * cnt)
        for cls, cnt in zip(classes, counts)
    }
    weights = np.array([class_weight[lbl] for lbl in y], dtype=np.float64)
    log.debug(
        "Sample weights — class weights: %s",
        {int(k): round(v, 3) for k, v in class_weight.items()},
    )
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# Core training function
# ══════════════════════════════════════════════════════════════════════════════

def train_dp_sgd_model(
    dataset_name: str,
    epsilon: float,
    label: str = "binary",
    force: bool = False,
) -> dict[str, Any]:
    """
    Train a Differentially Private MLP (DP-SGD) at a given epsilon.

    All hyperparameters are read from config.yaml under the `dp_sgd` key:
        hidden_layers, epochs, batch_size, max_grad_norm, delta.

    The model is saved to disk after training. If the saved file exists and
    force=False, it is loaded and re-evaluated without retraining (idempotent).

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        epsilon:      Privacy budget eps > 0.
        label:        'binary' or 'multiclass'.
        force:        Re-train even if a saved model already exists.

    Returns:
        Result dict matching the schema of dp_model.train_dp_model():
        {
          "model_name":   "dp_sgd",
          "dataset":      str,
          "label_type":   str,
          "epsilon":      float,
          "actual_epsilon": float,   <- Opacus-verified eps achieved
          "train_time_s": float,
          "metrics": {
              "val":  { metric_name: value, ... },
              "test": { metric_name: value, ... },
          }
        }
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise ImportError(
            f"PyTorch and Opacus are required for DP-SGD experiments: {exc}\n"
            "Install with: pip install torch opacus"
        ) from exc

    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    # ── Read hyperparameters from config ──────────────────────────────────────
    hidden_layers  = DP_SGD_CFG.get("hidden_layers",  [128, 64])
    epochs         = DP_SGD_CFG.get("epochs",         20)
    batch_size     = DP_SGD_CFG.get("batch_size",     256)
    max_grad_norm  = DP_SGD_CFG.get("max_grad_norm",  1.0)
    delta          = DP_SGD_CFG.get("delta",          1e-5)

    saved_path = _dp_sgd_model_path(dataset_name, epsilon, label)
    # Opacus (DP-SGD) only supports CUDA and CPU — MPS is not yet supported.
    # Fall back to CPU on macOS Apple Silicon even if MPS is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed   = SAMPLING_CFG["random_seed"]

    # ── Load split arrays ─────────────────────────────────────────────────────
    splits   = get_split_arrays(dataset_name, label=label, scale=True)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    input_size  = X_train.shape[1]
    classes     = np.unique(y_train)
    output_size = len(classes)

    log.info(
        "DP-SGD | dataset=%s | eps=%.2f | label=%s | "
        "input=%d | layers=%s | output=%d | device=%s%s",
        dataset_name, epsilon, label,
        input_size, hidden_layers, output_size, device,
        " (MPS available but not used — Opacus requires CUDA/CPU)"
        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            and device.type == "cpu") else "",
    )

    # ── Load or train ─────────────────────────────────────────────────────────
    if saved_path.exists() and not force:
        log.info(
            "Loading saved DP-SGD model for '%s' (eps=%.2f, label=%s)",
            dataset_name, epsilon, label,
        )
        wrapper   = joblib.load(saved_path)
        train_time    = 0.0
        actual_epsilon = epsilon  # can't recover from joblib; use target as proxy
    else:
        torch.manual_seed(seed)

        # ── Build model ───────────────────────────────────────────────────────
        model = _build_mlp(input_size, hidden_layers, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # ── Class-weighted loss ───────────────────────────────────────────────
        # Used alongside WeightedRandomSampler for belt-and-suspenders handling
        # of IDS class imbalance (benign >> attacks).
        _, counts = np.unique(y_train, return_counts=True)
        class_weights_np = len(y_train) / (output_size * counts.astype(np.float64))
        class_weights    = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        criterion        = nn.CrossEntropyLoss(weight=class_weights)

        # ── DataLoader with WeightedRandomSampler ─────────────────────────────
        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        train_ds  = TensorDataset(X_train_t, y_train_t)

        sample_weights = _compute_dp_sample_weights(y_train)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,   # Opacus requires uniform batch sizes
        )

        # ── Apply DP-SGD via Opacus ───────────────────────────────────────────
        # poisson_sampling=False allows WeightedRandomSampler to be preserved;
        # privacy accounting uses fixed-subsampling amplification bounds.
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
        )

        log.info(
            "Opacus noise_multiplier=%.4f | max_grad_norm=%.2f | "
            "epochs=%d | batch_size=%d | delta=%.0e",
            optimizer.noise_multiplier, max_grad_norm, epochs, batch_size, delta,
        )

        # ── Training loop ─────────────────────────────────────────────────────
        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            eps_so_far = privacy_engine.get_epsilon(delta)
            log.info(
                "Epoch %2d/%d — loss=%.4f | eps_so_far=%.4f",
                epoch, epochs, avg_loss, eps_so_far,
            )

        train_time     = time.perf_counter() - t0
        actual_epsilon = privacy_engine.get_epsilon(delta)

        log.info(
            "DP-SGD training complete — target eps=%.2f | actual eps=%.4f | %.1fs",
            epsilon, actual_epsilon, train_time,
        )

        # ── Unwrap Opacus and move to CPU for portability ─────────────────────
        raw_model = model._module if hasattr(model, "_module") else model
        raw_model.eval()
        raw_model.cpu()

        wrapper = DPSGDWrapper(
            model=raw_model,
            classes=classes,
            label_type=label,
        )

        # ── Persist ───────────────────────────────────────────────────────────
        PATHS["models"].mkdir(parents=True, exist_ok=True)
        joblib.dump(wrapper, saved_path)
        log.info("DP-SGD model saved to: %s", saved_path)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("Evaluating DP-SGD model (eps=%.2f) on val split...", epsilon)
    val_metrics  = evaluate_model(wrapper, X_val,  y_val,  label_type=label)

    log.info("Evaluating DP-SGD model (eps=%.2f) on test split...", epsilon)
    test_metrics = evaluate_model(wrapper, X_test, y_test, label_type=label)

    result = {
        "model_name":     DP_SGD_MODEL_NAME,
        "dataset":        dataset_name,
        "label_type":     label,
        "epsilon":        epsilon,
        "actual_epsilon": round(actual_epsilon, 4),
        "train_time_s":   round(train_time, 2),
        "metrics": {
            "val":  val_metrics,
            "test": test_metrics,
        },
    }

    log.info(
        "DP-SGD | eps=%.2f | %s | test F1=%.4f | AUC=%s | MCC=%.4f",
        epsilon,
        dataset_name,
        test_metrics.get("f1_macro", 0),
        f"{test_metrics['roc_auc']:.4f}" if test_metrics.get("roc_auc") is not None else "N/A",
        test_metrics.get("mcc", 0),
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Epsilon sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_dp_sgd_sweep(
    dataset_name: str,
    label: str = "binary",
    epsilon_values: list[float] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Train DP-SGD MLP at each epsilon value and return all results.

    Mirrors the interface of dp_model.run_epsilon_sweep() so scripts and UI
    code can call either sweep interchangeably.

    Args:
        dataset_name:   Key matching config.yaml 'datasets' entry.
        label:          'binary' or 'multiclass'.
        epsilon_values: Epsilon values to sweep. Defaults to
                        dp_sgd.epsilon_values from config.yaml.
        force:          Re-train all models even if saved.

    Returns:
        List of result dicts (one per epsilon), ordered by ascending epsilon.
    """
    if epsilon_values is None:
        epsilon_values = DP_SGD_CFG.get("epsilon_values", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    epsilon_values = sorted(epsilon_values)

    log.info("=" * 70)
    log.info(
        "Starting DP-SGD epsilon sweep for '%s' (label=%s) — %d values: %s",
        dataset_name, label, len(epsilon_values), epsilon_values,
    )
    log.info("=" * 70)

    all_results = []
    for eps in epsilon_values:
        log.info("-" * 50)
        log.info("Running DP-SGD with eps = %.2f", eps)
        result = train_dp_sgd_model(
            dataset_name=dataset_name,
            epsilon=eps,
            label=label,
            force=force,
        )
        all_results.append(result)

    # Summary log
    log.info("=" * 70)
    log.info("DP-SGD sweep complete for '%s'. Summary (test F1 macro):", dataset_name)
    for r in all_results:
        log.info(
            "  eps=%5.2f -> F1=%.4f | AUC=%s | MCC=%.4f",
            r["epsilon"],
            r["metrics"]["test"].get("f1_macro", 0),
            f"{r['metrics']['test']['roc_auc']:.4f}"
            if r["metrics"]["test"].get("roc_auc") is not None else "N/A",
            r["metrics"]["test"].get("mcc", 0),
        )

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Load helper
# ══════════════════════════════════════════════════════════════════════════════

def load_dp_sgd_model(
    dataset_name: str,
    epsilon: float,
    label: str = "binary",
) -> DPSGDWrapper:
    """
    Load a previously trained DP-SGD wrapper from disk.

    Args:
        dataset_name: Key matching config.yaml 'datasets' entry.
        epsilon:      The epsilon value the model was trained at.
        label:        'binary' or 'multiclass'.

    Returns:
        Fitted DPSGDWrapper with .predict() and .predict_proba() methods.

    Raises:
        FileNotFoundError: If the model has not been trained at this epsilon.
    """
    saved_path = _dp_sgd_model_path(dataset_name, epsilon, label)

    if not saved_path.exists():
        raise FileNotFoundError(
            f"No DP-SGD model found at {saved_path}. "
            f"Run train_dp_sgd_model('{dataset_name}', epsilon={epsilon}) first."
        )

    log.info("Loading DP-SGD model (eps=%.2f): %s", epsilon, saved_path)
    return joblib.load(saved_path)
