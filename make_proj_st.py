"""DFPJ-проекция из HF safetensors базовой драфт-карты (для --dump-proj).

Берёт fc.weight [hidden, 8*hidden] и hidden_norm.weight [hidden] из
model.safetensors (incoai HF-чекпоинт) - это ровно то, что
DFlashDraftModel.forward применяет к target_hidden (заморожено в
тренировке по пути jzinno).

python make_proj_st.py <base_dir/model.safetensors> <out.dfpj>
"""
import json
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

src, dst = sys.argv[1], sys.argv[2]

with safe_open(src, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    fc_key = next(k for k in keys if k == "fc.weight" or k.endswith(".fc.weight"))
    norm_key = next(k for k in keys if k == "hidden_norm.weight")
    fc = f.get_tensor(fc_key).float().numpy()   # [out=2048, in=16384]
    norm = f.get_tensor(norm_key).float().numpy()  # [2048]

out_dim, in_dim = fc.shape
eps = 1e-6

with open(dst, "wb") as out:
    out.write(b"DFPJ")
    out.write(struct.pack("<III", 1, in_dim, out_dim))
    out.write(struct.pack("<f", eps))
    out.write(np.ascontiguousarray(fc).tobytes())
    out.write(np.ascontiguousarray(norm).tobytes())

print(f"DFPJ: in={in_dim} out={out_dim} eps={eps:g} "
      f"({fc_key}, {norm_key}) -> {dst}")
