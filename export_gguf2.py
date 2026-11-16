"""R3.4 (v3): стриминговый экспорт GGUF — сырой KV-копий + правильная ориентация.

Закрытые грабли (хроника в PROBLEMA-EXPORT.md):
1. ОРИЕНТАЦИЯ (фатально для v1/v2): авто-транспозиция «по сверке размерностей»
   была ошибкой. Шаблон хранит веса в стандартной ориентации GGUF:
   dims файла = reversed(torch shape) (ne0 = внутренняя ось), а данные —
   байты HF C-порядка КАК ЕСТЬ (проверено по всем 96 тензорам шаблона tier1;
   подтверждено gguf_writer: ti.shape[::-1] и issue ggml-org/llama.cpp#6040).
   Транспонировать не нужно НИЧЕГО; ассерт сверяет форму с шаблоном.
2. ТОКЕНАЙЗЕР копируется сырыми байтами из шаблона (никакой распаковки):
   decode/encode('utf-8','replace') лосси для байтовых BPE-токенов,
   а поэлементная упаковка массивов (248k+247k+248k) медленна и жрёт RAM.
3. n_kv считается точно при обходе (bool = 1 байт по спеке, не 4:
   в шаблоне есть BOOL-массив dflash.attention.sliding_window_pattern,
   на нём v2-обходчик терял синхронизацию -> 24-байтовый файл-труп).
4. F32-тензоры шаблона остаются F32 (нормы, conv_base); квантованные -> F16.
5. GGUF.* не пишем (это просто KV, лоадер не требует), general.name/file_type
   заменяем своими, остальной general.* сохраняем (провенанс).
6. Прогресс-лог по каждому тензору + валидация результата в конце
   (GGUFReader-паритет dims/типов, побайтовая сверка F32-нормы с шаблоном,
   dequant-спот fc.weight против HF-базы).

Память O(одного тензора) (~260 МБ пик). Время ~минута.

python export_gguf2.py <tier1-шаблон.gguf> <out-BF16.gguf>
"""
import mmap
import os
import struct
import sys
import time

import numpy as np
import torch
from gguf import GGUFReader, GGUFValueType as VT
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "train", "base")
LAYERS = os.path.join(HERE, "train", "out", "ru_draft_layers.pt")

template, out_path = sys.argv[1], sys.argv[2]
ALIGN = 32  # в шаблоне нет general.alignment -> дефолт по спеке
F32, F16 = 0, 1  # ggml type id

# размеры значений по спеке (bool = 1 байт!)
VT_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
           10: 8, 11: 8, 12: 8}

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
    "candidate_selector.successor_codebook": "selector_successor.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
    "norm.weight": "output_norm.weight",
}
N_LAYERS = 6

# --- план тензоров: gguf-имя -> hf-ключ ---
hf_of = {}
for hf, g in MAP.items():
    if "{n}" in hf:
        for i in range(N_LAYERS):
            hf_of[g.format(n=i)] = hf.format(n=i)
    else:
        hf_of[g] = hf


def skip_value(buf, pos, vt):
    if vt == 8:  # STRING
        (slen,) = struct.unpack_from("<Q", buf, pos)
        return pos + 8 + slen
    if vt == 9:  # ARRAY
        (it,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        (cnt,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        for _ in range(cnt):
            pos = skip_value(buf, pos, it)
        return pos
    return pos + VT_SIZE[vt]


# --- KV-секция шаблона: разбор на записи (имя, начало, конец) ---
with open(template, "rb") as ftmpl:
    mb = mmap.mmap(ftmpl.fileno(), 0, access=mmap.ACCESS_READ)
    if mb[:4] != b"GGUF":
        raise SystemExit("шаблон не GGUF")
    (ver,) = struct.unpack_from("<I", mb, 4)
    (n_tensors, n_kv) = struct.unpack_from("<QQ", mb, 8)
    pos = 24
    kv_entries = []
    for _ in range(n_kv):
        start = pos
        (klen,) = struct.unpack_from("<Q", mb, pos)
        pos += 8
        name = mb[pos:pos + klen].decode("utf-8")
        pos += klen
        (vt,) = struct.unpack_from("<I", mb, pos)
        pos += 4
        pos = skip_value(mb, pos, vt)
        kv_entries.append((name, start, pos))
    kept = [(n, s, e) for (n, s, e) in kv_entries
            if not n.startswith("GGUF.") and n not in ("general.file_type",
                                                       "general.name")]
    n_kv_out = len(kept) + 2
    print(f"KV шаблона: {n_kv} записей, копируем {len(kept)}, "
          f"+general.name/file_type -> n_kv={n_kv_out}")

    def pack_kv_str(name, val):
        nb = name.encode("utf-8")
        vb = val.encode("utf-8")
        return struct.pack("<Q", len(nb)) + nb + struct.pack("<I", 8) \
            + struct.pack("<Q", len(vb)) + vb

    def pack_kv_u32(name, val):
        nb = name.encode("utf-8")
        return struct.pack("<Q", len(nb)) + nb + struct.pack("<I", 4) \
            + struct.pack("<I", val)

    extra_kv = pack_kv_str("general.name", "Qwen3.6-35B-A3B-DFlash2-ru1") \
        + pack_kv_u32("general.file_type", 1)  # MOSTLY_F16
    mb.close()

# --- тензор-инфо шаблона (имена/dims/типы) через GGUFReader ---
rdr = GGUFReader(template)
tinfo = {t.name: (tuple(int(x) for x in t.shape), int(t.tensor_type))
         for t in rdr.tensors}
missing = set(tinfo) - set(hf_of)
extra = set(hf_of) - set(tinfo)
assert not missing and not extra, f"план vs шаблон: -{missing} +{extra}"

# --- тензор-инфо блока (dims шаблона дословно; тип = F32 если шаблон F32) ---
order = [t.name for t in rdr.tensors]  # порядок шаблона
ti_out = bytearray()
offsets = {}
off = 0
for gname in order:
    shape, ttype = tinfo[gname]
    otype = F32 if ttype == F32 else F16
    nelem = int(np.prod(shape))
    nbytes = nelem * (4 if otype == F32 else 2)
    offsets[gname] = (off, otype, nbytes)
    nb = gname.encode("utf-8")
    ti_out += struct.pack("<Q", len(nb)) + nb
    ti_out += struct.pack("<I", len(shape))
    for d in shape:
        ti_out += struct.pack("<Q", d)
    ti_out += struct.pack("<I", otype)
    ti_out += struct.pack("<Q", off)
    off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
total_data = off
print(f"тензоров: {len(order)}, данных: {total_data/1e6:.0f} МБ")

# --- запись файла ---
t0 = time.time()
delta = torch.load(LAYERS, map_location="cpu", weights_only=True) \
    if os.path.exists(LAYERS) else {}
n_delta = 0
with open(out_path, "wb") as out, \
        safe_open(os.path.join(BASE, "model.safetensors"),
                  framework="pt", device="cpu") as sf:
    out.write(b"GGUF")
    out.write(struct.pack("<I", 3))
    out.write(struct.pack("<QQ", len(order), n_kv_out))
    # KV: сырые байты шаблона + наши два ключа
    with open(template, "rb") as fraw:
        for name, s, e in kept:
            fraw.seek(s)
            remain = e - s
            while remain:
                chunk = fraw.read(min(remain, 1 << 22))
                if not chunk:
                    raise SystemExit("EOF при копировании KV " + name)
                out.write(chunk)
                remain -= len(chunk)
    out.write(extra_kv)
    out.write(bytes(ti_out))
    pad = (ALIGN - out.tell() % ALIGN) % ALIGN
    out.write(b"\x00" * pad)
    # данные: по одному тензору, HF C-порядок БЕЗ транспозиции
    for i, gname in enumerate(order):
        hkey = hf_of[gname]
        _, otype, nbytes = offsets[gname]
        want_shape = tinfo[gname][0]  # dims файла = reversed(torch shape)
        t = delta.get(hkey)
        if t is not None:
            n_delta += 1
        else:
            t = sf.get_tensor(hkey)  # ленивый view: конвертим внутри with!
        ts = tuple(int(x) for x in t.shape)
        assert ts == tuple(reversed(want_shape)), \
            f"{gname}: torch {ts} != reversed(шаблон) {want_shape}"
        if otype == F32:
            arr = t.float().contiguous().numpy().reshape(-1)
        else:
            t16 = t.half()
            assert not torch.isinf(t16).any(), f"{gname}: bf16->f16 overflow"
            arr = t16.contiguous().numpy().reshape(-1)
        assert arr.nbytes == nbytes, f"{gname}: {arr.nbytes} != {nbytes}"
        out.write(arr.tobytes())
        pad = (ALIGN - nbytes % ALIGN) % ALIGN
        out.write(b"\x00" * pad)
        del t, arr
        if (i + 1) % 12 == 0 or i + 1 == len(order):
            el = time.time() - t0
            done = i + 1
            eta = el / done * (len(order) - done)
            print(f"  [{done}/{len(order)}] {gname} "
                  f"({nbytes/1e6:.0f} МБ, {el:.0f}s, ETA {eta:.0f}s)")
print(f"наложено обученных: {n_delta}/{len(order)}; "
      f"файл {os.path.getsize(out_path)/1e6:.0f} МБ за {time.time()-t0:.0f}s")

# --- валидация ---
r2 = GGUFReader(out_path)
td2 = {t.name: t for t in r2.tensors}
td1 = {t.name: t for t in rdr.tensors}
assert len(r2.tensors) == len(order), "число тензоров"
for t in r2.tensors:
    shape, ttype = tinfo[t.name]
    assert tuple(int(x) for x in t.shape) == shape, f"dims {t.name}"
    want_t = F32 if ttype == F32 else F16
    assert int(t.tensor_type) == want_t, f"тип {t.name}"
assert "tokenizer.ggml.model" in r2.fields, "нет tokenizer.ggml.model"
n_tok = len(list(r2.fields["tokenizer.ggml.tokens"].data))
assert n_tok == 248320, f"tokens {n_tok}"
# F32-норма (заморожена, источник тот же HF) должна совпасть байт-в-байт
a = td2["enc.output_norm.weight"].data.tobytes()
b = td1["enc.output_norm.weight"].data.tobytes()
assert a == b, "enc.output_norm.weight: F32-норма не совпала с шаблоном"
# dequant-спот: fc.weight (Q6_K в шаблоне) против HF-базы
tq = td1["fc.weight"]
flat = np.ascontiguousarray(tq.data, dtype=np.uint8).reshape(-1)
blk_sz, type_sz = GGML_QUANT_SIZES[tq.tensor_type]
n_elem = 2048
n_bytes = (n_elem // blk_sz) * type_sz  # целые блоки
assert n_bytes > 0
with safe_open(os.path.join(BASE, "model.safetensors"), framework="pt",
               device="cpu") as sf:
    hf_fc = sf.get_tensor("fc.weight")[:1, :n_elem].float().numpy()
dq = dequantize(flat[:n_bytes], tq.tensor_type)
assert dq.size == n_elem, f"dequant size {dq.size} != {n_elem}"
corr = float(np.corrcoef(dq, hf_fc[0])[0, 1])
print(f"валидация: KV {len(r2.fields)}, tokens {n_tok}, тензоры "
      f"{len(r2.tensors)}, норма=шаблон ок, fc dequant corr={corr:.4f}")
assert corr > 0.99, "fc dequant-корреляция с HF провалилась"
print(f"ГОТОВО: {out_path}")
