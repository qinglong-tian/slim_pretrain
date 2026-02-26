"""Simplified prior generation for the new PU pretraining pipeline.

Key differences vs the legacy pipeline:
- Raw size is derived from (P, Eta, pi) rather than sampled directly.
- Train/test ordering is class-stratified.
- PU train side is built as labeled positives first, then dropped rows.
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
    "sign": SignActivation,
    "heaviside": HeavisideActivation,
    "rbf": RBFActivation,
    "sine": SineActivation,
    "square": SquareActivation,
    "abs": AbsActivation,
}

_DIFFICULTY_PRESETS = {
    "easy": {"noise_std": 0.005, "num_layers": 2, "hidden_dim": 16},
    "medium": {"noise_std": 0.01, "num_layers": 3, "hidden_dim": 32},
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


def _assign_labels_by_ratio(score: Tensor, class1_ratio: float) -> Tensor:
    """Assign binary labels with exact class-1 count based on score ranking."""
    n = int(score.numel())
    n1 = int(round(float(class1_ratio) * n))
    n1 = max(0, min(n1, n))

    y = torch.zeros(n, dtype=torch.long, device=score.device)
    if n1 == 0:
        return y
    if n1 == n:
        return torch.ones_like(y)

    top_idx = torch.argsort(score, descending=True)[:n1]
    y[top_idx] = 1
    return y


def _sample_num_categorical_features(
    num_features: int,
    ratio_range: Tuple[float, float],
    rng: np.random.Generator,
) -> int:
    lo_ratio, hi_ratio = ratio_range
    lo_count = int(np.floor(float(lo_ratio) * float(num_features)))
    hi_count = int(np.ceil(float(hi_ratio) * float(num_features)))
    lo_count = int(np.clip(lo_count, 0, num_features))
    hi_count = int(np.clip(hi_count, lo_count, num_features))
    if hi_count == 0:
        return 0
    return int(rng.integers(lo_count, hi_count + 1))


def _randomly_discretize_features(
    X: Tensor,
    train_rows: int,
    cfg: "SimplifiedPriorConfig",
    rng: np.random.Generator,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Randomly convert a subset of features to categorical bins."""
    num_features = int(X.shape[1])
    feature_is_categorical = torch.zeros((num_features,), dtype=torch.bool, device=X.device)
    feature_cardinalities = torch.zeros((num_features,), dtype=torch.long, device=X.device)

    num_categorical = _sample_num_categorical_features(
        num_features=num_features,
        ratio_range=cfg.categorical_feature_ratio_range,
        rng=rng,
    )
    if num_categorical <= 0:
        return X, feature_is_categorical, feature_cardinalities

    sampled = rng.choice(num_features, size=num_categorical, replace=False)
    train_rows = int(max(1, min(train_rows, int(X.shape[0]))))
    X_out = X.clone()
    card_lo, card_hi = cfg.categorical_cardinality_range

    for feature_index in sampled:
        requested_cardinality = int(rng.integers(card_lo, card_hi + 1))
        train_slice = X_out[:train_rows, int(feature_index)].contiguous()
        quantile_grid = torch.linspace(
            0.0,
            1.0,
            requested_cardinality + 1,
            dtype=X_out.dtype,
            device=X_out.device,
        )[1:-1]
        boundaries = torch.quantile(train_slice, quantile_grid) if quantile_grid.numel() > 0 else train_slice[:0]
        boundaries = torch.unique(boundaries)

        if boundaries.numel() == 0:
            discretized = torch.zeros_like(X_out[:, int(feature_index)], dtype=torch.long)
            actual_cardinality = 2
        else:
            feature_values = X_out[:, int(feature_index)].contiguous()
            discretized = torch.bucketize(feature_values, boundaries=boundaries)
            actual_cardinality = max(2, int(boundaries.numel()) + 1)
            discretized = discretized.clamp(min=0, max=actual_cardinality - 1)

        X_out[:, int(feature_index)] = discretized.to(dtype=X_out.dtype)
        feature_is_categorical[int(feature_index)] = True
        feature_cardinalities[int(feature_index)] = int(actual_cardinality)

    return X_out, feature_is_categorical, feature_cardinalities


@dataclass
class SimplifiedPriorConfig:
    # Dataset size / split
    seq_len: int = 1024
    train_size: float | int = 0.7
    min_test_size: int = 1
    split_strategy: str = "stratified"

    # PU composition controls.
    # Note: with hard-drop batching, the hidden pool is removed before model input.
    positive_train_size: Optional[int] = 160
    unlabeled_to_positive_ratio: float = 1.0
    test_class1_ratio: float = 0.5

    # Raw generated class proportion control
    class1_ratio: float = 0.5

    # MLP-SCM structure
    num_features: int = 20
    num_causes: int = 20
    num_layers: int = 3
    hidden_dim: int = 32

    # Optional causal-graph flavored construction (TabICL-style).
    is_causal: bool = False
    noncausal_feature_source: str = "head"  # head | roots
    y_is_effect: bool = True
    in_clique: bool = False
    sort_features: bool = True
    categorical_feature_ratio_range: Tuple[float, float] = (0.0, 0.0)
    categorical_cardinality_range: Tuple[int, int] = (2, 10)

    # Nonlinearity family used by the prior.
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
        if len(self.categorical_feature_ratio_range) != 2:
            raise ValueError("categorical_feature_ratio_range must contain exactly two values.")
        cat_ratio_lo = float(self.categorical_feature_ratio_range[0])
        cat_ratio_hi = float(self.categorical_feature_ratio_range[1])
        if not (0.0 <= cat_ratio_lo <= cat_ratio_hi <= 1.0):
            raise ValueError("categorical_feature_ratio_range must satisfy 0 <= lo <= hi <= 1.")
        if len(self.categorical_cardinality_range) != 2:
            raise ValueError("categorical_cardinality_range must contain exactly two values.")
        card_lo = int(self.categorical_cardinality_range[0])
        card_hi = int(self.categorical_cardinality_range[1])
        if card_lo < 2 or card_lo > card_hi:
            raise ValueError("categorical_cardinality_range must satisfy 2 <= lo <= hi.")
        if not self.is_causal and self.noncausal_feature_source == "roots":
            if int(self.num_causes) != int(self.num_features):
                raise ValueError(
                    "When is_causal=False and noncausal_feature_source='roots', require num_causes == num_features."
                )

        if self.split_strategy not in {"stratified"}:
            raise ValueError("split_strategy must be 'stratified'.")
        if not (0.0 <= float(self.class1_ratio) <= 1.0):
            raise ValueError("class1_ratio must be in [0, 1].")
        if not (0.0 <= float(self.test_class1_ratio) < 1.0):
            raise ValueError("test_class1_ratio must be in [0, 1).")
        if float(self.unlabeled_to_positive_ratio) <= 0.0:
            raise ValueError("unlabeled_to_positive_ratio must be > 0.")
        if int(self.min_test_size) < 1:
            raise ValueError("min_test_size must be >= 1.")

        self._ensure_causal_capacity()

    def _ensure_causal_capacity(self) -> None:
        if not self.is_causal:
            return
        needed = int(self.num_features) + 1
        blocks = max(int(self.num_layers) - 1, 1)
        min_hidden_dim = int(np.ceil(needed / blocks))
        if self.hidden_dim < min_hidden_dim:
            self.hidden_dim = min_hidden_dim

    def resolve_positive_train_size(self) -> int:
        if self.positive_train_size is not None:
            p = int(self.positive_train_size)
            if p < 1:
                raise ValueError("positive_train_size must be >= 1 when provided.")
            return p

        # Fallback only when positive_train_size is unset:
        # derive P from train_size and test_class1_ratio.
        if isinstance(self.train_size, float):
            if not (0.0 < self.train_size < 1.0):
                raise ValueError("If train_size is float, it must be in (0, 1).")
            t = int(round(self.seq_len * self.train_size))
        else:
            t = int(self.train_size)
        p = int(round((1.0 - float(self.test_class1_ratio)) * t))
        return max(1, p)

    def resolve_test_size(self, positive_train_size: int) -> int:
        s = int(round(float(positive_train_size) * float(self.unlabeled_to_positive_ratio)))
        if s < 1:
            raise ValueError("Resolved test size must be >= 1.")
        return s

    def resolve_pre_pu_train_size(self, positive_train_size: int) -> int:
        denom = 1.0 - float(self.test_class1_ratio)
        if denom <= 0.0:
            raise ValueError("test_class1_ratio must satisfy pi < 1.")
        t = int(round(float(positive_train_size) / denom))
        if t < int(positive_train_size):
            t = int(positive_train_size)
        return t

    def resolve_seq_len(self, positive_train_size: int) -> int:
        return int(self.resolve_pre_pu_train_size(positive_train_size) + self.resolve_test_size(positive_train_size))


class SimpleMLPSCMPrior(nn.Module):
    """Minimal MLP-SCM prior."""

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
                act_name = str(np.random.choice(list(cfg.nonlinearities)))
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
        y = _assign_labels_by_ratio(score=score, class1_ratio=float(cfg.class1_ratio))
        return X.float(), y.long()

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


def _apply_structured_pu_hiding(
    y: Tensor,
    cfg: SimplifiedPriorConfig,
    rng: np.random.Generator,
) -> Tuple[Tensor, Tensor, Tensor, int, bool, int, int, Dict[str, float]]:
    """Create a class-stratified PU split from (P, Eta, pi)-derived sizes with hard drop."""
    y_true = y.clone().long()
    n = int(y_true.numel())

    positive_train_size = int(cfg.resolve_positive_train_size())
    train_size = int(cfg.resolve_pre_pu_train_size(positive_train_size))
    test_size = int(cfg.resolve_test_size(positive_train_size))
    expected_seq_len = int(train_size + test_size)

    if n != expected_seq_len:
        raise ValueError(
            "Raw dataset size must match derived size from (P, Eta, pi). "
            f"Got seq_len={n}, expected={expected_seq_len}."
        )
    if test_size < int(cfg.min_test_size):
        raise ValueError(
            "Test split too small for stable training. "
            f"Got test_size={test_size}, required >= {cfg.min_test_size}."
        )

    n1 = int((y_true == 1).sum().item())
    n0 = n - n1
    requested_test_ratio = float(np.clip(float(cfg.test_class1_ratio), 0.0, 1.0))

    lower_train_class1 = max(0, n1 - test_size)
    upper_train_class1 = min(n1, train_size, train_size - positive_train_size)
    if lower_train_class1 > upper_train_class1:
        raise ValueError(
            "No feasible class-stratified split with the requested (P, Eta, pi). "
            f"Bounds for train_class1 are [{lower_train_class1}, {upper_train_class1}]."
        )

    target_train_class1 = int(round(requested_test_ratio * float(train_size)))
    train_class1 = int(np.clip(target_train_class1, lower_train_class1, upper_train_class1))
    train_class0 = train_size - train_class1
    test_class1 = n1 - train_class1
    test_class0 = test_size - test_class1

    if train_class0 > n0:
        raise ValueError(
            "Not enough class-0 rows for requested positive-size split. "
            f"Need {train_class0}, have {n0}."
        )
    if test_class0 < 0 or test_class1 < 0:
        raise ValueError("Invalid test class allocation.")

    idx0 = torch.where(y_true == 0)[0].cpu().numpy()
    idx1 = torch.where(y_true == 1)[0].cpu().numpy()
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    train0 = idx0[:train_class0]
    test0 = idx0[train_class0 : train_class0 + test_class0]
    train1 = idx1[:train_class1]
    test1 = idx1[train_class1 : train_class1 + test_class1]

    if train0.shape[0] != train_class0 or train1.shape[0] != train_class1:
        raise RuntimeError("Insufficient rows for requested train class allocation.")
    if test0.shape[0] != test_class0 or test1.shape[0] != test_class1:
        raise RuntimeError("Insufficient rows for requested test class allocation.")

    train0 = train0.copy()
    rng.shuffle(train0)
    pos_idx = train0[:positive_train_size]
    dropped_idx = np.concatenate([train0[positive_train_size:], train1])
    rng.shuffle(dropped_idx)

    test_idx = np.concatenate([test0, test1])
    rng.shuffle(test_idx)

    ordered_np = np.concatenate([pos_idx, dropped_idx, test_idx])
    if int(ordered_np.shape[0]) != n:
        raise RuntimeError("Internal split bug: row count mismatch after stratified ordering.")

    ordered_idx = torch.from_numpy(ordered_np).to(dtype=torch.long, device=y_true.device)
    y_out = y_true[ordered_idx].clone()
    label_observed_mask = torch.ones_like(y_out, dtype=torch.bool)

    # Hard-drop policy: always hide non-positive train rows.
    is_pu = True
    removed_class = 1
    removed_class_original = 1
    y_out[positive_train_size:train_size] = -1
    label_observed_mask[positive_train_size:train_size] = False

    train_true = y_true[ordered_idx[:train_size]]
    test_true = y_true[ordered_idx[train_size:]]
    dropped_true = y_true[ordered_idx[positive_train_size:train_size]]

    realized_train_ratio = float(train_true.float().mean().item()) if train_size > 0 else 0.0
    realized_test_ratio = float(test_true.float().mean().item()) if test_size > 0 else 0.0
    realized_dropped_ratio = float(dropped_true.float().mean().item()) if dropped_true.numel() > 0 else 0.0

    meta = {
        "positive_train_size": float(positive_train_size),
        "test_size": float(test_size),
        "requested_unlabeled_to_positive_ratio": float(cfg.unlabeled_to_positive_ratio),
        "requested_test_class1_ratio": requested_test_ratio,
        "realized_train_class1_ratio": realized_train_ratio,
        "realized_test_class1_ratio": realized_test_ratio,
        "realized_unlabeled_class1_ratio": realized_dropped_ratio,
        "raw_class1_ratio": float(y_true.float().mean().item()),
    }

    return (
        ordered_idx,
        y_out,
        label_observed_mask,
        train_size,
        is_pu,
        removed_class,
        removed_class_original,
        meta,
    )


def generate_simplified_prior_data(
    cfg: SimplifiedPriorConfig,
    num_datasets: int = 1,
) -> Dict[str, Tensor]:
    """Generate one or more simplified prior datasets."""
    if num_datasets < 1:
        raise ValueError("num_datasets must be >= 1.")

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    Xs = []
    ys = []
    label_observed_masks = []
    train_sizes = []
    seq_lens = []
    is_pu_flags = []
    removed_classes = []
    removed_classes_original = []

    positive_train_sizes = []
    test_sizes = []
    requested_test_class1_ratios = []
    realized_train_class1_ratios = []
    realized_test_class1_ratios = []
    realized_unlabeled_class1_ratios = []
    raw_class1_ratios = []
    feature_is_categorical = []
    feature_cardinalities = []

    for _ in range(num_datasets):
        prior = SimpleMLPSCMPrior(cfg)
        with torch.no_grad():
            X, y_true = prior()

        rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
        (
            ordered_idx,
            y,
            label_observed_mask,
            train_size,
            is_pu,
            removed_class,
            removed_class_original,
            meta,
        ) = _apply_structured_pu_hiding(y=y_true, cfg=cfg, rng=rng)

        X = X[ordered_idx]
        X, is_categorical_i, cardinalities_i = _randomly_discretize_features(
            X=X,
            train_rows=train_size,
            cfg=cfg,
            rng=rng,
        )

        Xs.append(X.detach())
        ys.append(y.detach())
        label_observed_masks.append(label_observed_mask.detach())
        train_sizes.append(int(train_size))
        seq_lens.append(int(y.shape[0]))
        is_pu_flags.append(bool(is_pu))
        removed_classes.append(int(removed_class))
        removed_classes_original.append(int(removed_class_original))

        positive_train_sizes.append(int(meta["positive_train_size"]))
        test_sizes.append(int(meta["test_size"]))
        requested_test_class1_ratios.append(float(meta["requested_test_class1_ratio"]))
        realized_train_class1_ratios.append(float(meta["realized_train_class1_ratio"]))
        realized_test_class1_ratios.append(float(meta["realized_test_class1_ratio"]))
        realized_unlabeled_class1_ratios.append(float(meta["realized_unlabeled_class1_ratio"]))
        raw_class1_ratios.append(float(meta["raw_class1_ratio"]))
        feature_is_categorical.append(is_categorical_i.detach())
        feature_cardinalities.append(cardinalities_i.detach())

    X = torch.stack(Xs, dim=0).cpu()
    y = torch.stack(ys, dim=0).cpu()
    label_observed_mask = torch.stack(label_observed_masks, dim=0).cpu()

    return {
        "X": X,
        "y": y,
        "label_observed_mask": label_observed_mask,
        "train_sizes": torch.tensor(train_sizes, dtype=torch.long),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "is_pu": torch.tensor(is_pu_flags, dtype=torch.bool),
        "removed_class": torch.tensor(removed_classes, dtype=torch.long),
        "removed_class_original": torch.tensor(removed_classes_original, dtype=torch.long),
        "positive_train_sizes": torch.tensor(positive_train_sizes, dtype=torch.long),
        "test_sizes": torch.tensor(test_sizes, dtype=torch.long),
        "requested_test_class1_ratio": torch.tensor(requested_test_class1_ratios, dtype=torch.float32),
        "realized_train_class1_ratio": torch.tensor(realized_train_class1_ratios, dtype=torch.float32),
        "realized_test_class1_ratio": torch.tensor(realized_test_class1_ratios, dtype=torch.float32),
        "realized_unlabeled_class1_ratio": torch.tensor(realized_unlabeled_class1_ratios, dtype=torch.float32),
        "raw_class1_ratio": torch.tensor(raw_class1_ratios, dtype=torch.float32),
        "feature_is_categorical": torch.stack(feature_is_categorical, dim=0).cpu(),
        "feature_cardinalities": torch.stack(feature_cardinalities, dim=0).cpu(),
    }


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
