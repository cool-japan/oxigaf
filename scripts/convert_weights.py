#!/usr/bin/env python3
"""Convert GAF / ImageDream PyTorch weights to SafeTensors for Candle loading.

Usage:
    python convert_weights.py <checkpoint.pt> <output_dir/>

Requires: torch, safetensors

Scope note: this is a one-time, offline asset-conversion step, not part of
the OxiGAF Pure Rust runtime. It exists because PyTorch `.pt` checkpoints
are pickle-serialized; reading them requires `torch.load`, which is why this
script depends on Python + PyTorch. The OxiGAF crates themselves (oxigaf,
oxigaf-bridge, etc.) have no Python or C/C++ dependency in their default
build — that claim is about the Rust runtime, not about how you get a
third-party `.pt` checkpoint into `.safetensors` form in the first place.
`oxigaf-bridge` currently reads/writes `.safetensors` only; it does not yet
ingest raw `.pt`/`.pkl` pickle files, so this script (and
`convert_flame.py` for `.pkl` FLAME models) is still required for that
first-mile conversion. Once you have `.safetensors` output, the rest of the
pipeline is Pure Rust.
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
