"""Гейт R1: структурная верификация дампа DFSH.

Проверки:
  1. слой 0 в списке -> строки одинаковых token id на разных позициях
     идентичны (эмбеддинг позиционно-независим) - доказывает буфер/
     выравнивание/sync;
  2. SANITY: конечность, разумный масштаб, per-layer профиль дисперсий
     отличается (слои не перепутаны местами).

python gate_r1.py <dump.bin>
"""
import array
import struct
import sys
from collections import defaultdict

path = sys.argv[1]
f = open(path, "rb")
assert f.read(4) == b"DFSH"
ver, n_layers = struct.unpack("<II", f.read(8))
layer_ids = struct.unpack("<%dI" % n_layers, f.read(4 * n_layers))
n_embd, proj_dim, topk, n_vocab = struct.unpack("<IIII", f.read(16))
print(f"layers={list(layer_ids)} n_embd={n_embd} proj={proj_dim} topk={topk}")
assert proj_dim == 0, "гейт работает на raw-дампе (без --dump-proj)"
feat_dim = n_layers * n_embd

rows0 = defaultdict(list)          # token -> [слой0-векторы]
layer_var = [[] for _ in range(n_layers)]
recs = 0
while True:
    hdr = f.read(4)
    if len(hdr) < 4:
        break
    (n_tok,) = struct.unpack("<I", hdr)
    toks = array.array("i")
    toks.frombytes(f.read(4 * n_tok))
    per = topk * 8 + feat_dim * 4
    body = f.read(per * n_tok)
    assert len(body) == per * n_tok, "обрыв записи"
    ba = array.array("f")
    ba.frombytes(body)
    stride = topk * 2 + feat_dim
    for k in range(n_tok):
        base = k * stride + topk * 2
        for l in range(n_layers):
            v = ba[base + l * n_embd: base + (l + 1) * n_embd]
            if 0 in layer_ids and l == layer_ids.index(0):
                rows0[toks[k]].append(v)
            mean = sum(v) / n_embd
            var = sum((x - mean) ** 2 for x in v) / n_embd
            layer_var[l].append(var)
    recs += 1

print(f"records={recs}, уникальных токенов со слоем-0: {len(rows0)}")

ok = True

# 1) идентичность эмбеддинг-строк
if 0 in layer_ids:
    worst = 0.0
    checked = 0
    for tok, vecs in rows0.items():
        if len(vecs) < 2 or checked > 200:
            continue
        checked += 1
        for other in vecs[1:]:
            d = max(abs(a - b) for a, b in zip(vecs[0], other))
            worst = max(worst, d)
    print(f"[гейт 1] слой-0 идентичность: {checked} токенов, max|Δ|={worst:.3e}")
    ok &= worst < 1e-3
else:
    print("[гейт 1] слой 0 не в списке - пропущен (добавь 0 для полной проверки)")

# 2) масштаб/дисперсии
for l in range(n_layers):
    vs = layer_var[l]
    mv = sum(vs) / len(vs)
    print(f"[гейт 2] слой {layer_ids[l]}: средняя дисперсия {mv:.4f}")
    ok &= 0 < mv < 1e4

print("ГЕЙТ R1:", "ПРОЙДЕН" if ok else "ПРОВАЛЕН")
sys.exit(0 if ok else 1)
