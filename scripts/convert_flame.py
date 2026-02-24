#!/usr/bin/env python3
"""Convert a FLAME .pkl model to a directory of .npy files for Rust loading.

Usage:
    python convert_flame.py <input.pkl> <output_dir/>

Requires: numpy, scipy, pickle (stdlib)
"""

import pickle
import sys
from pathlib import Path

import numpy as np


def convert_flame(input_pkl: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_pkl} ...")
    with open(input_pkl, "rb") as f:
        model = pickle.load(f, encoding="latin1")

    # --- Template vertices ---
    v_template = np.array(model["v_template"], dtype=np.float32)  # [5023, 3]
    np.save(output / "v_template.npy", v_template)

    # --- Faces ---
    faces = np.array(model["f"], dtype=np.int32)  # [9976, 3]
    np.save(output / "faces.npy", faces)

    # --- Blend shapes ---
    # FLAME packs shape (300) + expression (100) into shapedirs [5023, 3, 400]
    shapedirs_full = np.array(model["shapedirs"], dtype=np.float32)
    n_total = shapedirs_full.shape[2]
    n_shape = min(300, n_total)
    n_expr = n_total - n_shape

    shapedirs = shapedirs_full[:, :, :n_shape]  # [5023, 3, 300]
    np.save(output / "shapedirs.npy", shapedirs)

    if n_expr > 0:
        expressiondirs = shapedirs_full[:, :, n_shape:]  # [5023, 3, 100]
    else:
        # Some FLAME versions store expression dirs separately
        expressiondirs = np.zeros((v_template.shape[0], 3, 0), dtype=np.float32)
    np.save(output / "expressiondirs.npy", expressiondirs)

    # --- Pose blend shapes ---
    posedirs = np.array(model["posedirs"], dtype=np.float32)  # [5023, 3, 36]
    np.save(output / "posedirs.npy", posedirs)

    # --- Joint regressor (sparse → dense) ---
    j_regressor = model["J_regressor"]
    try:
        import scipy.sparse

        if scipy.sparse.issparse(j_regressor):
            j_regressor = np.array(j_regressor.todense(), dtype=np.float32)
        else:
            j_regressor = np.array(j_regressor, dtype=np.float32)
    except ImportError:
        j_regressor = np.array(j_regressor, dtype=np.float32)
    np.save(output / "j_regressor.npy", j_regressor)

    # --- Kinematic tree ---
    kintree_table = np.array(model["kintree_table"], dtype=np.int32)  # [2, 5]
    np.save(output / "kintree_table.npy", kintree_table)

    # --- LBS weights ---
    lbs_weights = np.array(model["weights"], dtype=np.float32)  # [5023, 5]
    np.save(output / "lbs_weights.npy", lbs_weights)

    # --- Summary ---
    print(f"Saved to {output}/")
    for name, arr in [
        ("v_template", v_template),
        ("faces", faces),
        ("shapedirs", shapedirs),
        ("expressiondirs", expressiondirs),
        ("posedirs", posedirs),
        ("j_regressor", j_regressor),
        ("kintree_table", kintree_table),
        ("lbs_weights", lbs_weights),
    ]:
        print(f"  {name:20s}  {str(arr.shape):20s}  {arr.dtype}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pkl> <output_dir/>")
        sys.exit(1)
    convert_flame(sys.argv[1], sys.argv[2])
