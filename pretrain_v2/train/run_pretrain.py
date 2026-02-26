from __future__ import annotations

from pathlib import Path

from . import PretrainConfig, default_base_prior_config, pretrain_nano_tabpfn_pu


def main() -> None:
    base_cfg = default_base_prior_config()
    cfg = PretrainConfig()
    legacy_init = Path(__file__).resolve().parents[1] / "saved_models" / "legacy_model.pt"
    init_from = str(legacy_init) if legacy_init.exists() else None
    if init_from is not None:
        print(f"Using default legacy init checkpoint: {init_from}")
    result = pretrain_nano_tabpfn_pu(base_cfg=base_cfg, config=cfg, init_from=init_from)
    hist = result["history"]
    print(f"Finished {len(hist)} steps.")
    print("Last record:", hist[-1] if len(hist) > 0 else None)


if __name__ == "__main__":
    main()
