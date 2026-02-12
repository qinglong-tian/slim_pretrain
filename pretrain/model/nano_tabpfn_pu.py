from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
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
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdim=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdim=True) + 1e-20
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
    """scikit-learn-like interface for the PU-adapted model."""

    def __init__(self, model: NanoTabPFNPUModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.num_classes = 2

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y_t = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x_t, y_t), train_test_split_index=len(self.X_train)).squeeze(0)
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_test).argmax(axis=1)
