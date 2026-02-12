# Slim Pretraining Package

This folder is a self-contained copy of the pretraining pipeline.

Included components:
- Prior generation (`simplified_prior`): MLP-SCM binary prior with PU handling.
- Data curriculum: stage-wise sampling of `is_causal`, `num_layers`, `hidden_dim`.
- Batch generation: variable-size padded batches for transformer pretraining.
- Model: PU-adapted NanoTabPFN architecture.
- Training curriculum: warmup + cosine learning-rate schedule.

Prior behavior note:
- `SimplifiedPriorConfig.noncausal_feature_source` controls non-causal feature generation.
- Use `"head"` (default) for `x_head(h)` features, or `"roots"` for TabICL-like root-cause features.
- For `"roots"`, require `num_causes == num_features`.

Run a default pretraining job:

```bash
python -m slim_pretrain.pretrain.train.run_pretrain
```

Or from Python:

```python
from slim_pretrain import default_base_prior_config, PretrainConfig, pretrain_nano_tabpfn_pu

base_cfg = default_base_prior_config()
cfg = PretrainConfig()
out = pretrain_nano_tabpfn_pu(base_cfg=base_cfg, config=cfg)
```
