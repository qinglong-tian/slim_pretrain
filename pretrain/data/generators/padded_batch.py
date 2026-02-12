from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from slim_pretrain.simplified_prior import SimplifiedPriorConfig, generate_simplified_prior_data


@dataclass(frozen=True)
class VariableBatchSpec:
    batch_size: int
    seq_len_range: Tuple[int, int]
    num_features_range: Tuple[int, int]
    train_ratio_range: Tuple[float, float] = (0.5, 0.8)
    num_causes_mode: str = "equal_features"  # equal_features | fixed
    fixed_num_causes: Optional[int] = None
    pu_row_policy: str = "drop"  # drop | keep_masked


def _validate_spec(spec: VariableBatchSpec) -> None:
    if spec.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    seq_lo, seq_hi = spec.seq_len_range
    feat_lo, feat_hi = spec.num_features_range
    tr_lo, tr_hi = spec.train_ratio_range

    if seq_lo < 2 or seq_lo > seq_hi:
        raise ValueError("seq_len_range must satisfy 2 <= min <= max.")
    if feat_lo < 1 or feat_lo > feat_hi:
        raise ValueError("num_features_range must satisfy 1 <= min <= max.")
    if not (0.0 < tr_lo < tr_hi < 1.0):
        raise ValueError("train_ratio_range must satisfy 0 < min < max < 1.")

    if spec.num_causes_mode not in {"equal_features", "fixed"}:
        raise ValueError("num_causes_mode must be 'equal_features' or 'fixed'.")
    if spec.num_causes_mode == "fixed":
        if spec.fixed_num_causes is None or spec.fixed_num_causes < 1:
            raise ValueError("fixed_num_causes must be set to >= 1 when num_causes_mode='fixed'.")
    if spec.pu_row_policy not in {"drop", "keep_masked"}:
        raise ValueError("pu_row_policy must be 'drop' or 'keep_masked'.")


def _sample_num_causes(spec: VariableBatchSpec, num_features: int) -> int:
    if spec.num_causes_mode == "equal_features":
        return int(num_features)
    return int(spec.fixed_num_causes)  # validated in _validate_spec


def _sample_cfg(
    base_cfg: SimplifiedPriorConfig,
    spec: VariableBatchSpec,
    rng: np.random.Generator,
) -> SimplifiedPriorConfig:
    cfg_dict = asdict(base_cfg)
    cfg_dict["difficulty"] = None

    seq_len = int(rng.integers(spec.seq_len_range[0], spec.seq_len_range[1] + 1))
    num_features = int(rng.integers(spec.num_features_range[0], spec.num_features_range[1] + 1))
    train_ratio = float(rng.uniform(spec.train_ratio_range[0], spec.train_ratio_range[1]))

    cfg_dict["seq_len"] = seq_len
    cfg_dict["num_features"] = num_features
    cfg_dict["num_causes"] = _sample_num_causes(spec, num_features)
    cfg_dict["train_size"] = train_ratio

    # Each dataset uses a different seed while preserving reproducibility via rng.
    cfg_dict["seed"] = int(rng.integers(0, 2**31 - 1))
    return SimplifiedPriorConfig(**cfg_dict)


def generate_variable_padded_batch(
    base_cfg: SimplifiedPriorConfig,
    spec: VariableBatchSpec,
    rng: Optional[np.random.Generator] = None,
    pad_value_x: float = 0.0,
    pad_value_y: int = -1,
    config_sampler: Optional[Callable[[np.random.Generator], SimplifiedPriorConfig]] = None,
) -> Dict[str, Tensor]:
    """Generate a batch of variable-size datasets and pad to shared shape.

    Returns:
    - X: (B, T_max, F_max) float
    - y: (B, T_max) long
    - row_mask: (B, T_max) bool
    - feature_mask: (B, F_max) bool
    - train_mask: (B, T_max) bool
    - test_mask: (B, T_max) bool
    - label_observed_mask: (B, T_max) bool
    - train_labeled_mask: (B, T_max) bool
    - train_unlabeled_mask: (B, T_max) bool
    - original_train_sizes: (B,) long
    - removed_train_rows: (B,) long
    - seq_lens: (B,) long
    - num_features: (B,) long
    - train_sizes: (B,) long
    - is_pu: (B,) bool
    - removed_class: (B,) long
    - removed_class_original: (B,) long
    - cfg_is_causal: (B,) bool
    - cfg_num_layers: (B,) long
    - cfg_hidden_dim: (B,) long
    - cfg_pu_keep_probability: (B,) float
    """
    _validate_spec(spec)
    if rng is None:
        rng = np.random.default_rng()

    datasets = []
    seq_lens = []
    num_features = []
    train_sizes = []
    original_train_sizes = []
    removed_train_rows = []
    cfg_is_causal = []
    cfg_num_layers = []
    cfg_hidden_dim = []
    cfg_pu_keep_probability = []

    for _ in range(spec.batch_size):
        base_cfg_i = config_sampler(rng) if config_sampler is not None else base_cfg
        cfg_i = _sample_cfg(base_cfg=base_cfg_i, spec=spec, rng=rng)
        out_i = generate_simplified_prior_data(cfg_i, num_datasets=1)
        Xi = out_i["X"][0]
        yi = out_i["y"][0]
        li = out_i["label_observed_mask"][0]
        ti = int(out_i["train_sizes"][0].item())
        is_pu_i = bool(out_i["is_pu"][0].item())
        removed_class_i = int(out_i["removed_class"][0].item())
        removed_class_original_i = int(out_i["removed_class_original"][0].item())

        original_ti = ti
        removed_rows_i = 0
        if spec.pu_row_policy == "drop":
            # Remove train rows with hidden PU labels before transformer batching.
            train_rows = torch.zeros_like(li, dtype=torch.bool)
            train_rows[:ti] = True
            drop_rows = train_rows & ~li
            removed_rows_i = int(drop_rows.sum().item())
            keep_rows = ~drop_rows
            Xi = Xi[keep_rows]
            yi = yi[keep_rows]
            li = li[keep_rows]
            ti = int(original_ti - removed_rows_i)

        datasets.append((Xi, yi, li, ti, is_pu_i, removed_class_i, removed_class_original_i))
        seq_lens.append(int(Xi.shape[0]))
        num_features.append(int(Xi.shape[1]))
        train_sizes.append(ti)
        original_train_sizes.append(original_ti)
        removed_train_rows.append(removed_rows_i)
        cfg_is_causal.append(bool(cfg_i.is_causal))
        cfg_num_layers.append(int(cfg_i.num_layers))
        cfg_hidden_dim.append(int(cfg_i.hidden_dim))
        cfg_pu_keep_probability.append(float(cfg_i.pu_keep_probability))

    batch_size = spec.batch_size
    max_rows = max(seq_lens)
    max_features = max(num_features)

    X_pad = torch.full((batch_size, max_rows, max_features), float(pad_value_x), dtype=torch.float32)
    y_pad = torch.full((batch_size, max_rows), int(pad_value_y), dtype=torch.long)
    row_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)
    feature_mask = torch.zeros((batch_size, max_features), dtype=torch.bool)
    train_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)
    test_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)
    label_observed_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)
    train_labeled_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)
    train_unlabeled_mask = torch.zeros((batch_size, max_rows), dtype=torch.bool)
    is_pu = torch.zeros((batch_size,), dtype=torch.bool)
    removed_class = torch.full((batch_size,), -1, dtype=torch.long)
    removed_class_original = torch.full((batch_size,), -1, dtype=torch.long)

    for idx, (Xi, yi, li, ti, is_pu_i, removed_class_i, removed_class_original_i) in enumerate(datasets):
        rows, feats = Xi.shape
        X_pad[idx, :rows, :feats] = Xi
        y_pad[idx, :rows] = yi.long()
        row_mask[idx, :rows] = True
        feature_mask[idx, :feats] = True
        train_mask[idx, :ti] = True
        test_mask[idx, ti:rows] = True
        label_observed_mask[idx, :rows] = li.bool()
        train_labeled_mask[idx, :rows] = train_mask[idx, :rows] & label_observed_mask[idx, :rows]
        train_unlabeled_mask[idx, :rows] = train_mask[idx, :rows] & ~label_observed_mask[idx, :rows]
        is_pu[idx] = is_pu_i
        removed_class[idx] = removed_class_i
        removed_class_original[idx] = removed_class_original_i

    return {
        "X": X_pad,
        "y": y_pad,
        "row_mask": row_mask,
        "feature_mask": feature_mask,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "label_observed_mask": label_observed_mask,
        "train_labeled_mask": train_labeled_mask,
        "train_unlabeled_mask": train_unlabeled_mask,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "num_features": torch.tensor(num_features, dtype=torch.long),
        "train_sizes": torch.tensor(train_sizes, dtype=torch.long),
        "original_train_sizes": torch.tensor(original_train_sizes, dtype=torch.long),
        "removed_train_rows": torch.tensor(removed_train_rows, dtype=torch.long),
        "is_pu": is_pu,
        "removed_class": removed_class,
        "removed_class_original": removed_class_original,
        "cfg_is_causal": torch.tensor(cfg_is_causal, dtype=torch.bool),
        "cfg_num_layers": torch.tensor(cfg_num_layers, dtype=torch.long),
        "cfg_hidden_dim": torch.tensor(cfg_hidden_dim, dtype=torch.long),
        "cfg_pu_keep_probability": torch.tensor(cfg_pu_keep_probability, dtype=torch.float32),
    }
