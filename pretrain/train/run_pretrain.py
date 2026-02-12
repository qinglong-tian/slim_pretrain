from __future__ import annotations

from slim_pretrain.pretrain.train import PretrainConfig, default_base_prior_config, pretrain_nano_tabpfn_pu


def main() -> None:
    base_cfg = default_base_prior_config()
    cfg = PretrainConfig()
    result = pretrain_nano_tabpfn_pu(base_cfg=base_cfg, config=cfg)
    hist = result["history"]
    print(f"Finished {len(hist)} steps.")
    print("Last record:", hist[-1] if len(hist) > 0 else None)


if __name__ == "__main__":
    main()
