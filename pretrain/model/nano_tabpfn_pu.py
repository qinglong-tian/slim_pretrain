from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Union
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention


class NanoTabPFNPUModel(nn.Module):
    """PU-adapted NanoTabPFN-style model with 2-class output."""

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int = 2,
    ):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoderPU(embedding_size)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderLayerPU(
                    embedding_size=embedding_size,
                    nhead=num_attention_heads,
                    mlp_hidden_size=mlp_hidden_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)

        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src_table = torch.cat([x_src, y_src], dim=2)

        for block in self.transformer_blocks:
            src_table = block(src_table, train_test_split_index=train_test_split_index)

        output = src_table[:, train_test_split_index:, -1, :]
        return self.decoder(output)


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        train_rows = int(max(0, min(train_test_split_index, x.shape[1])))
        if train_rows >= 2:
            train_slice = x[:, :train_rows]
            mean = torch.mean(train_slice, dim=1, keepdim=True)
            std = torch.std(train_slice, dim=1, keepdim=True, unbiased=False).clamp_min(1e-20)
        elif train_rows == 1:
            train_slice = x[:, :1]
            mean = torch.mean(train_slice, dim=1, keepdim=True)
            # With a single labeled row, variance is undefined/degenerate; avoid scaling.
            std = torch.ones_like(mean)
        else:
            # No labeled rows: keep inputs unchanged by normalization.
            mean = torch.zeros_like(x[:, :1])
            std = torch.ones_like(x[:, :1])
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoderPU(nn.Module):
    """Encodes labels with a learnable token for unlabeled entries."""

    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)
        self.unlabeled_embedding = nn.Parameter(torch.zeros(1, 1, embedding_size))
        nn.init.normal_(self.unlabeled_embedding, std=0.02)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        if y_train.dim() == 2:
            y_train = y_train.unsqueeze(-1)
        if y_train.shape[1] > num_rows:
            raise ValueError("y_train rows exceed total num_rows.")

        batch_size = y_train.shape[0]
        pad_rows = num_rows - y_train.shape[1]
        if pad_rows > 0:
            padding = torch.full(
                (batch_size, pad_rows, 1),
                -1.0,
                dtype=y_train.dtype,
                device=y_train.device,
            )
            y_full = torch.cat([y_train, padding], dim=1)
        else:
            y_full = y_train

        observed_mask = y_full >= 0
        y_for_linear = y_full.clone()
        y_for_linear[~observed_mask] = 0.0
        embedded = self.linear_layer(y_for_linear)  # (B, R, E)

        unlabeled = self.unlabeled_embedding.expand(batch_size, num_rows, -1)
        embedded = torch.where(observed_mask.expand_as(embedded), embedded, unlabeled)
        return embedded.unsqueeze(2)  # (B, R, 1, E)


class TransformerEncoderLayerPU(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        # Attention between features.
        src_f = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src_f = self.self_attention_between_features(src_f, src_f, src_f)[0] + src_f
        src = src_f.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)

        # Attention between datapoints.
        src = src.transpose(1, 2)
        src_d = src.reshape(batch_size * col_size, rows_size, embedding_size)

        src_left = self.self_attention_between_datapoints(
            src_d[:, :train_test_split_index],
            src_d[:, :train_test_split_index],
            src_d[:, :train_test_split_index],
        )[0]
        # PU/semi-supervised change: test queries can attend to all datapoints.
        src_right = self.self_attention_between_datapoints(
            src_d[:, train_test_split_index:],
            src_d,
            src_d,
        )[0]
        src_d = torch.cat([src_left, src_right], dim=1) + src_d

        src = src_d.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)

        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabPFNPUClassifier:
    """scikit-learn-like interface for the PU-adapted model.

    This wrapper matches the pretraining `drop` PU policy:
    - Train labels are always the observed positive class (class 0).
    - `fit` therefore only needs train features.
    """

    def __init__(self, model: NanoTabPFNPUModel, device: Union[torch.device, str] = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.num_classes = 2
        self.X_train: Optional[np.ndarray] = None

    @staticmethod
    def default_checkpoint_path() -> Path:
        # Project root /checkpoints/latest.pt
        return Path(__file__).resolve().parents[2] / "checkpoints" / "latest.pt"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> "NanoTabPFNPUClassifier":
        if checkpoint_path is None:
            checkpoint_path = cls.default_checkpoint_path()
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Expected latest model at {cls.default_checkpoint_path()}."
            )
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint must be a dict-like payload: {checkpoint_path}")
        if "model_state_dict" not in payload:
            raise ValueError(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")

        model_cfg = payload.get("config", {})
        model_cfg = model_cfg.get("model", {}) if isinstance(model_cfg, dict) else {}
        model = NanoTabPFNPUModel(
            embedding_size=int(model_cfg.get("embedding_size", 128)),
            num_attention_heads=int(model_cfg.get("num_attention_heads", 8)),
            mlp_hidden_size=int(model_cfg.get("mlp_hidden_size", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_outputs=int(model_cfg.get("num_outputs", 2)),
        )
        model.load_state_dict(payload["model_state_dict"])
        return cls(model=model, device=device)

    def fit(self, X_train: np.ndarray):
        X_train_arr = np.asarray(X_train, dtype=np.float32)
        if X_train_arr.ndim != 2:
            raise ValueError(f"Expected X_train to be 2D, got shape {X_train_arr.shape}.")
        if X_train_arr.shape[0] == 0:
            raise ValueError("X_train must contain at least one row.")
        self.X_train = X_train_arr
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise RuntimeError("Classifier is not fitted. Call fit(X_train) before predict_proba.")
        X_test_arr = np.asarray(X_test, dtype=np.float32)
        if X_test_arr.ndim != 2:
            raise ValueError(f"Expected X_test to be 2D, got shape {X_test_arr.shape}.")
        if X_test_arr.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Feature mismatch: X_train has {self.X_train.shape[1]} features, "
                f"but X_test has {X_test_arr.shape[1]}."
            )

        x = np.concatenate((self.X_train, X_test_arr), axis=0)
        y_train = np.zeros((self.X_train.shape[0],), dtype=np.float32)
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y_t = torch.from_numpy(y_train).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x_t, y_t), train_test_split_index=len(self.X_train)).squeeze(0)
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_test).argmax(axis=1)
