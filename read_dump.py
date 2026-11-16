"""Читатель дампа DFSH (валидатор гейта R1 + загрузчик для тренировки).

Формат: magic "DFSH", u32 ver=1, u32 n_layers, u32 layer_ids[n],
u32 n_embd, u32 proj_dim (0 = raw), u32 topk, u32 n_vocab; далее записи:
  u32 n_tokens; i32 token_ids[n_tokens];
  per token: i32 topk_ids[topk]; f32 topk_logits[topk]; f32 feat[feat_dim]

python read_dump.py <dump.bin> [--limit N]   - сводка/санити
"""
import array
import struct
import sys

path = sys.argv[1]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10 ** 9

f = open(path, "rb")
assert f.read(4) == b"DFSH", "не DFSH"
ver, n_layers = struct.unpack("<II", f.read(8))
assert ver == 1
layer_ids = struct.unpack("<%dI" % n_layers, f.read(4 * n_layers))
n_embd, proj_dim, topk, n_vocab = struct.unpack("<IIII", f.read(16))
feat_dim = proj_dim if proj_dim else n_layers * n_embd

print(f"layers={list(layer_ids)} n_embd={n_embd} proj_dim={proj_dim} "
      f"topk={topk} n_vocab={n_vocab} feat_dim={feat_dim}")

recs = 0
tokens_total = 0
feat_min = feat_max = None
feat_abs_sum = 0.0
bad = 0
while recs < limit:
    hdr = f.read(4)
    if len(hdr) < 4:
        break
    (n_tok,) = struct.unpack("<I", hdr)
    toks = array.array("i")
    toks.frombytes(f.read(4 * n_tok))
    assert all(0 <= t < n_vocab for t in toks), "token id вне словаря"
    f.read(4 * topk * n_tok)   # блок top-k ids
    f.read(4 * topk * n_tok)   # блок top-k логитов
    per = feat_dim * 4
    body = f.read(per * n_tok)
    assert len(body) == per * n_tok, "обрыв записи"
    fa = array.array("f")
    fa.frombytes(body)
    for k in range(n_tok):
        feats = fa[k * feat_dim:(k + 1) * feat_dim]
        m = min(feats)
        M = max(feats)
        feat_min = m if feat_min is None else min(feat_min, m)
        feat_max = M if feat_max is None else max(feat_max, M)
        feat_abs_sum += sum(abs(v) for v in feats) / feat_dim
    tokens_total += n_tok
    recs += 1

print(f"records={recs} tokens={tokens_total}")
if tokens_total:
    print(f"feat: min={feat_min:.4f} max={feat_max:.4f} "
          f"mean|f|={feat_abs_sum / tokens_total:.4f}")
    if proj_dim:
        ok = feat_min == feat_min and feat_max == feat_max and 0.01 < feat_abs_sum / tokens_total < 100
    else:
        ok = -10 < feat_min < 0 < feat_max < 10 and 0.01 < feat_abs_sum / tokens_total < 100
    print("SANITY:", "OK" if ok else "ПОДОЗРИТЕЛЬНЫЕ ЗНАЧЕНИЯ (NaN/масштаб?)")
