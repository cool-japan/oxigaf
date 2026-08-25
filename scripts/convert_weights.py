#!/usr/bin/env python3
"""Convert GAF / ImageDream PyTorch weights to SafeTensors for Candle loading.

DEPRECATED as of 0.1.2 — superseded by a pure-Rust implementation.

    cargo run -p oxigaf-bridge --example convert_pytorch -- \
        --checkpoint <checkpoint.pt> --output-dir <output_dir/> --precision fp16

`oxigaf_bridge::convert_pytorch_checkpoint` now reads `.pt` checkpoints
directly: `oxigaf-bridge/src/pickle/` implements a non-executing pickle
reader plus the ZIP-container and tensor-rebuild handling a `torch.save`
file needs, so no Python, no PyTorch, and no `torch.load` is involved.
It reproduces this script's behaviour exactly — the same component prefixes,
the same partitioning into unet/vae/clip/other — with two improvements:
`--precision` is optional (this script always forced FP16), and
non-contiguous tensor views (e.g. a `.t()` transpose) are materialized in
row-major order rather than being copied verbatim.

That closes the last Python dependency in the pipeline: OxiGAF is now Pure
Rust end to end, including the first-mile asset conversion.

This script is kept as a reference and an escape hatch for exotic
checkpoints (e.g. the legacy pre-1.6 `torch.save` format, which is a bare
pickle rather than a ZIP archive and which the Rust reader rejects with an
explicit message). Nothing in the OxiGAF pipeline requires it.

Usage:
    python convert_weights.py <checkpoint.pt> <output_dir/>

Requires: torch, safetensors
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_weights(input_path: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    # If wrapped in a "state_dict" key, unwrap
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # --- Partition weights by component ---
    unet_weights = {}
    vae_weights = {}
    clip_weights = {}
    other_weights = {}

    for key, value in state_dict.items():
        tensor = value.contiguous().half()  # fp16

        if key.startswith("model.diffusion_model.") or key.startswith("unet."):
            clean = key.replace("model.diffusion_model.", "").replace("unet.", "")
            unet_weights[clean] = tensor
        elif key.startswith("first_stage_model.") or key.startswith("vae."):
            clean = key.replace("first_stage_model.", "").replace("vae.", "")
            vae_weights[clean] = tensor
        elif key.startswith("cond_stage_model.") or key.startswith("clip."):
            clean = key.replace("cond_stage_model.", "").replace("clip.", "")
            clip_weights[clean] = tensor
        else:
            other_weights[key] = tensor

    # --- Save as SafeTensors ---
    for name, weights in [
        ("unet", unet_weights),
        ("vae", vae_weights),
        ("clip", clip_weights),
        ("other", other_weights),
    ]:
        if weights:
            path = output / f"{name}.safetensors"
            save_file(weights, str(path))
            total_params = sum(t.numel() for t in weights.values())
            print(f"  {name}: {len(weights)} tensors, {total_params:,} params → {path}")
        else:
            print(f"  {name}: (empty, skipped)")

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> <output_dir/>")
        sys.exit(1)
    convert_weights(sys.argv[1], sys.argv[2])
