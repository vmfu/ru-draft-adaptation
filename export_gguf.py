"""R3.4: экспорт дообученного драфта в GGUF (BF16) по шаблону tier1.

Читает базовый HF-чекпоинт (base/) + слой обученных тензоров
(out/ru_draft_layers.pt), мапит имена HF->GGUF (по тензор-карте tier1,
авто-транспозиция по сверке размерностей), копирует KV метаданные
шаблона. Выход BF16 -> потом llama-quantize картой tier1.

python export_gguf.py <template-tier1.gguf> <out-BF16.gguf>
"""
import os
import sys

import numpy as np
import torch
from gguf import GGUFReader, GGUFWriter
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "train", "base")
LAYERS = os.path.join(HERE, "train", "out", "ru_draft_layers.pt")

template, out_path = sys.argv[1], sys.argv[2]

N_LAYERS = 6

MAP = {
    "layers.{n}.self_attn.q_proj.weight": "blk.{n}.attn_q.weight",
    "layers.{n}.self_attn.k_proj.weight": "blk.{n}.attn_k.weight",
    "layers.{n}.self_attn.v_proj.weight": "blk.{n}.attn_v.weight",
    "layers.{n}.self_attn.o_proj.weight": "blk.{n}.attn_output.weight",
    "layers.{n}.self_attn.q_norm.weight": "blk.{n}.attn_q_norm.weight",
    "layers.{n}.self_attn.k_norm.weight": "blk.{n}.attn_k_norm.weight",
    "layers.{n}.input_layernorm.weight": "blk.{n}.attn_norm.weight",
    "layers.{n}.post_attention_layernorm.weight": "blk.{n}.ffn_norm.weight",
    "layers.{n}.mlp.gate_proj.weight": "blk.{n}.ffn_gate.weight",
    "layers.{n}.mlp.up_proj.weight": "blk.{n}.ffn_up.weight",
    "layers.{n}.mlp.down_proj.weight": "blk.{n}.ffn_down.weight",
    "layers.{n}.attention_conv.base_kernel": "blk.{n}.attn_conv_base",
    "layers.{n}.attention_conv.kernel_projection.weight": "blk.{n}.attn_conv_proj.weight",
    "layers.{n}.mlp_conv.base_kernel": "blk.{n}.ffn_conv_base",
    "layers.{n}.mlp_conv.kernel_projection.weight": "blk.{n}.ffn_conv_proj.weight",
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "norm.weight": "output_norm.weight",
}

# --- грузим HF-тензора: база + обученная дельта ---
state = {}
with safe_open(os.path.join(BASE, "model.safetensors"), framework="pt", device="cpu") as f:
    for k in f.keys():
        state[k] = f.get_tensor(k).clone()
if os.path.exists(LAYERS):
    delta = torch.load(LAYERS, map_location="cpu", weights_only=True)
    n_over = 0
    for k, v in delta.items():
        key = k[len("model."):] if k.startswith("model.") else k
        if key in state:
            state[key] = v
            n_over += 1
    print(f"обученных тензоров наложено: {n_over}")
    del delta
    import gc; gc.collect()
else:
    print("ВНИМАНИЕ: дельты нет, экспорт базы как есть")

# --- шаблон: измерения тензоров ---
rdr = GGUFReader(template)
gguf_shape = {}
gguf_names = set()
for t in rdr.tensors:
    gguf_shape[t.name] = tuple(int(x) for x in t.shape)
    gguf_names.add(t.name)

writer = GGUFWriter(out_path, "dflash")

# --- копируем KV шаблона по GGUFValueType ---
from gguf import GGUFValueType as _VT
kv_ok = 0
for field in rdr.fields.values():
    name = field.name.decode() if isinstance(field.name, (bytes, np.bytes_)) else field.name
    if name.split('.')[0] in ('general', 'GGUF'):
        continue
    t0 = field.types[0]
    try:
        if t0 == _VT.ARRAY:
            it = field.types[1] if len(field.types) > 1 else None
            vals = []
            for v in field.data:
                if it in (_VT.STRING,) or isinstance(v, (bytes, np.bytes_)):
                    vals.append(bytes(v).decode('utf-8', 'replace'))
                elif it == _VT.FLOAT32:
                    vals.append(float(v))
                else:
                    vals.append(int(v))
            writer.add_array(name, vals); kv_ok += 1
        elif t0 == _VT.STRING:
            writer.add_string(name, bytes(field.parts[-1]).decode('utf-8', 'replace'))
        elif t0 == _VT.UINT32:
            writer.add_uint32(name, int(field.parts[-1][0]))
        elif t0 == _VT.INT32:
            writer.add_int32(name, int(field.parts[-1][0]))
        elif t0 == _VT.FLOAT32:
            writer.add_float32(name, float(field.parts[-1][0]))
        elif t0 == _VT.UINT64:
            writer.add_uint64(name, int(field.parts[-1][0]))
        elif t0 == _VT.BOOL:
            writer.add_bool(name, bool(field.parts[-1][0]))
        else:
            print('KV', name, 'неизвестный тип', t0)
    except Exception as e:
        print('KV', name, 'пропущен:', e)

print(f"KV добавлено: {kv_ok}")
writer.add_file_type(30)  # BF16

written = 0
missing = []
# селекторы первыми (память на пике в конце - их транспозиция 242МБ)
for hf, g in [('candidate_selector.predecessor_codebook', 'selector_predecessor.weight'),
              ('candidate_selector.successor_codebook', 'selector_successor.weight'),
              ('candidate_selector.hidden_projection.weight', 'selector_hidden.weight')]:
    if hf in state:
        arr = state.pop(hf).half().numpy()
        want = gguf_shape[g]
        if tuple(arr.shape) != want and tuple(arr.T.shape) == want:
            arr = np.ascontiguousarray(arr.T)
        writer.add_tensor(g, arr)
        del arr
        import gc; gc.collect()
        written += 1
for hf, g in MAP.items():
    for n in (list(range(N_LAYERS)) if "{n}" in hf else [None]):
        hkey = hf.format(n=n) if n is not None else hf
        gkey = g.format(n=n) if n is not None else g
        if hkey not in state or gkey not in gguf_names:
            if "blk." not in gkey:
                missing.append(gkey)
            continue
        arr = state.pop(hkey).half().numpy()
        want = gguf_shape[gkey]
        if tuple(arr.shape) != want:
            if tuple(arr.T.shape) == want:
                arr = np.ascontiguousarray(arr.T)
            elif arr.ndim == 2 and want[0] == arr.shape[0] * arr.shape[1]:
                arr = arr.reshape(want)
            else:
                raise RuntimeError(f"{gkey}: {arr.shape} vs {want} (и транспозиция не подошла)")
        writer.add_tensor(gkey, arr)
        written += 1
        del arr
        import gc; gc.collect()

print(f"тензоров записано: {written}; без пары: {missing}")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"GGUF BF16 готов: {out_path}")
