"""Self-contained slim pretraining package."""

from .pretrain.train import (
    DataCurriculumConfig,
    ModelConfig,
    OptimConfig,
    PretrainConfig,
    default_base_prior_config,
    pretrain_nano_tabpfn_pu,
    stage_index_from_step,
    warmup_cosine_lr,
)

__all__ = [
    "ModelConfig",
    "OptimConfig",
    "DataCurriculumConfig",
    "PretrainConfig",
    "default_base_prior_config",
    "warmup_cosine_lr",
    "stage_index_from_step",
    "pretrain_nano_tabpfn_pu",
]
