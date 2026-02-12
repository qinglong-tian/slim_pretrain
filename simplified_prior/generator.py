"""Simplified prior generation (MLP-SCM only, binary labels only).

This module provides a compact, standalone prior generator inspired by TabICL's
MLP-SCM pipeline, with a smaller and simpler nonlinearity family in the spirit
of TabPFN-style synthetic priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class SignActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 2 * (x >= 0.0).float() - 1.0


class HeavisideActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return (x >= 0.0).float()


class RBFActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-(x**2))


class SineActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class SquareActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x**2


class AbsActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.abs(x)


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
    # TabICL-derived simple nonlinearities (excluding Fourier/random-frequency activations).
    "sign": SignActivation,
    "heaviside": HeavisideActivation,
    "rbf": RBFActivation,
    "sine": SineActivation,
    "square": SquareActivation,
    "abs": AbsActivation,
}

_DIFFICULTY_PRESETS = {
    # Easier: less noise, shallower and narrower generator network.
    "easy": {"noise_std": 0.005, "num_layers": 2, "hidden_dim": 16},
    # Medium close to previous default behavior.
    "medium": {"noise_std": 0.01, "num_layers": 3, "hidden_dim": 32},
    # Harder: noisier and more expressive latent mapping.
    "hard": {"noise_std": 0.03, "num_layers": 5, "hidden_dim": 64},
}


def _standardize_clip(x: Tensor, clip_value: float = 20.0) -> Tensor:
    """Column-wise standardization with clipping for numerical stability."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    z = (x - mean) / std
    return z.clamp(min=-clip_value, max=clip_value)


def _make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


@dataclass
class SimplifiedPriorConfig:
    # Dataset size / split
    seq_len: int = 512
    train_size: float | int = 0.5
    # MLP-SCM structure
    num_features: int = 20
    num_causes: int = 20
    num_layers: int = 3
    hidden_dim: int = 32
    # Optional causal-graph flavored construction (TabICL-style).
    is_causal: bool = False
    # Behavior when is_causal=False:
    # - "head": use learned x_head(h) features (current default behavior)
    # - "roots": use sampled root causes directly as features (TabICL-like)
    noncausal_feature_source: str = "head"  # head | roots
    y_is_effect: bool = True
    in_clique: bool = False
    sort_features: bool = True
    # Nonlinearity family used by the prior. If per_layer_activation=False,
    # the generator uses nonlinearities[0] for every layer.
    nonlinearities: Sequence[str] = (
        "tanh",
        "relu",
        "gelu",
        "identity",
        "sign",
        "heaviside",
        "rbf",
        "sine",
        "square",
        "abs",
    )
    per_layer_activation: bool = False
    # Noise / sampling
    noise_std: float = 0.01
    init_std: float = 0.8
    sampling: str = "normal"  # normal | uniform
    # Binary label setup
    balanced_labels: bool = True
    # PU label hiding control:
    # with probability `pu_keep_probability`, keep full labels;
    # otherwise hide one randomly chosen class in train rows only.
    # Default is PU-always generation.
    pu_keep_probability: float = 0.0
    # Randomness
    seed: Optional[int] = None
    device: str = "cpu"
    # Optional one-knob difficulty preset.
    difficulty: Optional[str] = None

    def __post_init__(self) -> None:
        if self.difficulty is not None:
            key = self.difficulty.lower()
            if key not in _DIFFICULTY_PRESETS:
                raise ValueError(
                    f"Unknown difficulty '{self.difficulty}'. Choose from: {sorted(_DIFFICULTY_PRESETS.keys())}."
                )
            preset = _DIFFICULTY_PRESETS[key]
            self.noise_std = float(preset["noise_std"])
            self.num_layers = int(preset["num_layers"])
            self.hidden_dim = int(preset["hidden_dim"])

        if self.noncausal_feature_source not in {"head", "roots"}:
            raise ValueError("noncausal_feature_source must be one of: 'head', 'roots'.")
        if not self.is_causal and self.noncausal_feature_source == "roots":
            if int(self.num_causes) != int(self.num_features):
                raise ValueError(
                    "When is_causal=False and noncausal_feature_source='roots', require num_causes == num_features."
                )

        self._ensure_causal_capacity()
        if not (0.0 <= float(self.pu_keep_probability) <= 1.0):
            raise ValueError("pu_keep_probability must be in [0, 1].")

    def _ensure_causal_capacity(self) -> None:
        if not self.is_causal:
            return
        needed = int(self.num_features) + 1
        blocks = max(int(self.num_layers) - 1, 1)
        min_hidden_dim = int(np.ceil(needed / blocks))
        if self.hidden_dim < min_hidden_dim:
            self.hidden_dim = min_hidden_dim

    def resolve_train_size(self) -> int:
        if isinstance(self.train_size, float):
            if not (0.0 < self.train_size < 1.0):
                raise ValueError("If train_size is float, it must be in (0, 1).")
            t = int(self.seq_len * self.train_size)
        else:
            t = int(self.train_size)
        if not (0 < t < self.seq_len):
            raise ValueError("Resolved train size must satisfy 0 < train_size < seq_len.")
        return t


class SimpleMLPSCMPrior(nn.Module):
    """Minimal MLP-SCM prior.

    - Samples root causes C.
    - Applies stacked blocks: activation -> linear -> additive Gaussian noise.
    - If `is_causal=False`, builds score from final representation and X from either:
      x_head(h) or root causes (config-controlled).
    - If `is_causal=True`, samples X/score from intermediate hidden variables.
    - Builds binary y by thresholding the score.
    """

    def __init__(self, cfg: SimplifiedPriorConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.num_layers < 2:
            raise ValueError("num_layers must be >= 2.")
        if cfg.noise_std < 0:
            raise ValueError("noise_std must be >= 0.")
        if len(cfg.nonlinearities) == 0:
            raise ValueError("nonlinearities must be non-empty.")

        self.device = torch.device(cfg.device)
        self.input_layer = nn.Linear(cfg.num_causes, cfg.hidden_dim)

        blocks = []
        for _ in range(cfg.num_layers - 1):
            if cfg.per_layer_activation:
                act_name = np.random.choice(list(cfg.nonlinearities))
            else:
                act_name = cfg.nonlinearities[0]
            blocks.append(
                nn.Sequential(
                    _make_activation(act_name),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.x_head = nn.Linear(cfg.hidden_dim, cfg.num_features)
        self.y_head = nn.Linear(cfg.hidden_dim, 1)
        self.to(self.device)
        self._init_weights(cfg.init_std)

    def _init_weights(self, std: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _sample_causes(self) -> Tensor:
        cfg = self.cfg
        if cfg.sampling == "normal":
            return torch.randn(cfg.seq_len, cfg.num_causes, device=self.device)
        if cfg.sampling == "uniform":
            return torch.rand(cfg.seq_len, cfg.num_causes, device=self.device)
        raise ValueError("sampling must be one of: 'normal', 'uniform'.")

    def forward(self) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg
        causes = self._sample_causes()
        h = self.input_layer(causes)
        intermediates = []
        for block in self.blocks:
            h = block(h)
            if cfg.noise_std > 0:
                h = h + torch.randn_like(h) * cfg.noise_std
            if cfg.is_causal:
                intermediates.append(h)

        if cfg.is_causal:
            X, score = self._sample_X_and_score_from_intermediates(intermediates)
        else:
            if cfg.noncausal_feature_source == "roots":
                X = causes
            else:
                X = self.x_head(h)
            score = self.y_head(h).squeeze(-1)

        X = _standardize_clip(X)
        score = _standardize_clip(score.unsqueeze(-1)).squeeze(-1)

        if cfg.balanced_labels:
            # Median split gives near-balanced binary labels.
            y = (score > torch.median(score)).long()
        else:
            y = (score > 0).long()
        return X.float(), y

    def _sample_X_and_score_from_intermediates(self, intermediates: list[Tensor]) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg
        if len(intermediates) == 0:
            raise ValueError("is_causal=True requires at least one intermediate hidden block output.")

        pool = torch.cat(intermediates, dim=1)
        _, total_vars = pool.shape
        needed = int(cfg.num_features) + 1
        if total_vars < needed:
            raise ValueError(f"Not enough intermediate variables: have {total_vars}, need {needed}.")

        num_blocks = len(intermediates)
        block_width = int(cfg.hidden_dim)
        first_block = torch.arange(0, min(block_width, total_vars), device=pool.device)
        last_start = (num_blocks - 1) * block_width
        last_block = torch.arange(last_start, min(last_start + block_width, total_vars), device=pool.device)

        if cfg.y_is_effect and len(last_block) > 0:
            y_pool = last_block
        elif len(first_block) > 0:
            y_pool = first_block
        else:
            y_pool = torch.arange(0, total_vars, device=pool.device)
        y_idx = int(y_pool[torch.randint(0, len(y_pool), (1,), device=pool.device)].item())

        if cfg.in_clique:
            clique_size = int(cfg.num_features) + 1
            start_min = max(0, y_idx - (clique_size - 1))
            start_max = min(y_idx, total_vars - clique_size)
            start = int(torch.randint(start_min, start_max + 1, (1,), device=pool.device).item())
            clique_indices = torch.arange(start, start + clique_size, device=pool.device)
            x_indices = clique_indices[clique_indices != y_idx]
        else:
            candidates = torch.arange(0, total_vars, device=pool.device)
            candidates = candidates[candidates != y_idx]
            perm = candidates[torch.randperm(len(candidates), device=pool.device)]
            x_indices = perm[: int(cfg.num_features)]

        if cfg.sort_features:
            x_indices, _ = torch.sort(x_indices)

        X = pool[:, x_indices]
        score = pool[:, y_idx]
        return X, score


def generate_simplified_prior_data(
    cfg: SimplifiedPriorConfig,
    num_datasets: int = 1,
) -> Dict[str, Tensor]:
    """Generate one or more simplified prior datasets.

    Returns tensors with shapes:
    - X: (B, T, F)
    - y: (B, T)
    - label_observed_mask: (B, T)
    - train_sizes: (B,)
    - seq_lens: (B,)
    - is_pu: (B,)
    - removed_class: (B,)
    - removed_class_original: (B,)
    """
    if num_datasets < 1:
        raise ValueError("num_datasets must be >= 1.")

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    train_size = cfg.resolve_train_size()
    Xs = []
    ys = []
    label_observed_masks = []
    is_pu_flags = []
    removed_classes = []
    removed_classes_original = []
    for _ in range(num_datasets):
        prior = SimpleMLPSCMPrior(cfg)
        with torch.no_grad():
            X, y = prior()
        y, label_observed_mask, is_pu, removed_class, removed_class_original = _apply_pu_hiding(
            y=y, train_size=train_size, pu_keep_probability=float(cfg.pu_keep_probability)
        )
        Xs.append(X.detach())
        ys.append(y.detach())
        label_observed_masks.append(label_observed_mask.detach())
        is_pu_flags.append(is_pu)
        removed_classes.append(removed_class)
        removed_classes_original.append(removed_class_original)

    X = torch.stack(Xs, dim=0).cpu()
    y = torch.stack(ys, dim=0).cpu()
    label_observed_mask = torch.stack(label_observed_masks, dim=0).cpu()
    train_sizes = torch.full((num_datasets,), train_size, dtype=torch.long)
    seq_lens = torch.full((num_datasets,), cfg.seq_len, dtype=torch.long)
    is_pu = torch.tensor(is_pu_flags, dtype=torch.bool)
    removed_class = torch.tensor(removed_classes, dtype=torch.long)
    removed_class_original = torch.tensor(removed_classes_original, dtype=torch.long)
    return {
        "X": X,
        "y": y,
        "label_observed_mask": label_observed_mask,
        "train_sizes": train_sizes,
        "seq_lens": seq_lens,
        "is_pu": is_pu,
        "removed_class": removed_class,
        "removed_class_original": removed_class_original,
    }


def _apply_pu_hiding(
    y: Tensor, train_size: int, pu_keep_probability: float
) -> Tuple[Tensor, Tensor, bool, int, int]:
    """Hide one random class in train rows only for PU datasets.

    Bernoulli outcome:
    - 1 (keep): no class is removed
    - 0 (PU): remove one randomly chosen class from training labels only
    """
    y_out = y.clone().long()
    label_observed_mask = torch.ones_like(y_out, dtype=torch.bool)

    keep_full_labels = bool(np.random.rand() < pu_keep_probability)
    if keep_full_labels:
        # Keep original labels as-is for non-PU samples.
        return y_out, label_observed_mask, False, -1, -1

    train_labels = y_out[:train_size]
    train_classes = torch.unique(train_labels)
    if len(train_classes) == 0:
        return y_out, label_observed_mask, False, -1, -1

    removed_class_original = int(train_classes[int(np.random.randint(0, len(train_classes)))].item())
    remove_mask = torch.zeros_like(y_out, dtype=torch.bool)
    remove_mask[:train_size] = y_out[:train_size] == removed_class_original

    if not bool(remove_mask.any()):
        return y_out, label_observed_mask, False, -1, -1

    # Enforce PU label convention:
    # - observed training class is always class 0
    # - removed/outlier class is always class 1
    # If we removed original class 0, flip 0<->1 on all observed labels.
    if removed_class_original == 0:
        swap_mask = y_out >= 0
        y_out[swap_mask] = 1 - y_out[swap_mask]

    y_out[remove_mask] = -1
    label_observed_mask[remove_mask] = False
    removed_class = 1
    return y_out, label_observed_mask, True, removed_class, removed_class_original


def split_dataset(X: Tensor, y: Tensor, train_size: int) -> Dict[str, Tensor]:
    """Split one dataset tensor pair into train/test tensors."""
    return {
        "X_train": X[:train_size],
        "y_train": y[:train_size],
        "X_test": X[train_size:],
        "y_test": y[train_size:],
    }


def summarize_class_counts(y: Tensor) -> Dict[int, int]:
    """Return label counts for a 1D binary label tensor."""
    labels, counts = torch.unique(y.long(), return_counts=True)
    return {int(k.item()): int(v.item()) for k, v in zip(labels, counts)}


def available_nonlinearities() -> Iterable[str]:
    return tuple(sorted(_ACTIVATIONS.keys()))


def available_difficulties() -> Iterable[str]:
    return tuple(sorted(_DIFFICULTY_PRESETS.keys()))
