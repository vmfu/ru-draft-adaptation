"""Сборка side-файла проекции (DFPJ) для дампера из BF16 драфт-карты.

fc.weight и enc.output_norm.weight из GGUF драфта -> плоский f32-файл,
который llama-imatrix --dump-proj применяет on-the-fly:
  конкат входов слоёв -> fc -> RMSNorm(enc.output_norm).

Формат DFPJ (little-endian): magic "DFPJ", u32 ver=1, u32 in_dim,
u32 out_dim, f32 eps, f32 fc[out_dim*in_dim]
(fc row-major: y[o] = sum_i fc[o*in_dim+i]*x[i]), f32 norm[out_dim].

Запуск: python make_proj.py <draft-BF16.gguf> <out.dfpj>
"""
import array
import struct
import sys

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]

SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

f = open(src, "rb")
assert f.read(4) == b"GGUF", "не GGUF"
ver, nten, nkv = struct.unpack("<IQQ", f.read(20))


def rstr(file):
    (n,) = struct.unpack("<Q", file.read(8))
    return file.read(n).decode("utf-8", "replace")


def skip_val(file, t):
    if t == 8:
        rstr(file)
    elif t == 9:
        (et,) = struct.unpack("<I", file.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            skip_val(file, et)
    else:
        file.read(SIZE[t])


eps = 1e-6
want = {"fc.weight": None, "enc.output_norm.weight": None}

for _ in range(nkv):
    k = rstr(f)
    (t,) = struct.unpack("<I", f.read(4))
    if k == "dflash.attention.layer_norm_rms_epsilon" and t == 6:
        (eps,) = struct.unpack("<f", f.read(4))
    else:
        skip_val(f, t)

pos_raw = f.tell()
pos_al = pos_raw + (32 - pos_raw % 32) % 32


def probe(file, pos):
    file.seek(pos)
    hdr = file.read(8)
    if len(hdr) < 8:
        return None
    (n,) = struct.unpack("<Q", hdr)
    if n > 64:
        return None
    s = file.read(n).decode("utf-8", "replace")
    return s if s and all(32 <= ord(c) < 127 for c in s) else None


f.seek(pos_al if probe(f, pos_al) is not None else pos_raw)

offs = {}
dims = {}
types = {}
for _ in range(nten):
    name = rstr(f)
    (nd,) = struct.unpack("<I", f.read(4))
    d = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
    (tt,) = struct.unpack("<I", f.read(4))
    (off,) = struct.unpack("<Q", f.read(8))
    offs[name] = off
    dims[name] = d
    types[name] = tt

assert "fc.weight" in offs and types["fc.weight"] == 30, "fc.weight нет или не BF16"
assert types.get("enc.output_norm.weight") == 0, "enc.output_norm.weight нет или не F32"

out_dim, in_dim = dims["fc.weight"][0], dims["fc.weight"][1]
assert in_dim % dims["enc.output_norm.weight"][0] == 0

blob = open(src, "rb").read()

fc_bytes = blob[offs["fc.weight"]:offs["fc.weight"] + out_dim * in_dim * 2]
u16 = array.array("H")
u16.frombytes(fc_bytes)
u32 = array.array("I", [h << 16 for h in u16])
fc_f32 = array.array("f")
fc_f32.frombytes(u32.tobytes())

norm_f32 = array.array("f")
norm_f32.frombytes(blob[offs["enc.output_norm.weight"]:
                        offs["enc.output_norm.weight"] + out_dim * 4])

with open(dst, "wb") as out:
    out.write(b"DFPJ")
    out.write(struct.pack("<III", 1, in_dim, out_dim))
    out.write(struct.pack("<f", eps))
    out.write(fc_f32.tobytes())
    out.write(norm_f32.tobytes())

print(f"DFPJ готов: {dst}: in={in_dim} out={out_dim} eps={eps:g} "
      f"fc {len(fc_f32) * 4 / 1e6:.0f} МБ")
