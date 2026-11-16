"""Экстрактор словарных таблиц ствола из GGUF (для тренировки драфта).

token_embd (Q3_K_M) + output.weight (Q4_K_S) из MiniPlus деквантятся в
f16 и складываются в target_tables.npz. ПАРИТЕТ: llama.cpp в бою кормит
драфта именно этими (квантованными) таблицами - тренировка на них
соответствует прод-инференсу точнее, чем на BF16 из HF.

Запуск (из venv): python extract_tables.py <MiniPlus.gguf> <out.npz>
"""
import struct
import sys

import numpy as np

QK = 256


def deq_q3_k(raw, n_blocks):
    """hmask[32] qs[64] scales[12] d(f16); точный порт dequantize_row_q3_K."""
    out = np.empty(n_blocks * QK, dtype=np.float32)
    kmask1 = np.uint32(0x03030303)
    kmask2 = np.uint32(0x0F0F0F0F)
    for i in range(n_blocks):
        b = raw[i * 110:(i + 1) * 110]
        hmask, qs, scales, d = b[:32], b[32:96], b[96:108], b[108:110]
        d_all = np.frombuffer(d, dtype=np.float16)[0].astype(np.float32)
        aux = np.zeros(4, dtype=np.uint32)
        aux[:3] = np.frombuffer(scales, dtype=np.uint32)
        tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        sc = aux.view(np.uint8).view(np.int8)  # 16 signed масштабов
        y = np.empty(QK, dtype=np.float32)
        is_ = 0
        m = 1
        hm = np.frombuffer(hmask, dtype=np.uint8)
        qi = 0  # q += 32 после каждого n0
        for n0 in (0, 128):
            shift = 0
            q = np.frombuffer(qs[qi:qi + 32], dtype=np.uint8).astype(np.int32)
            for _ in range(4):
                for half in (0, 16):
                    dl = d_all * (int(sc[is_]) - 32)
                    is_ += 1
                    bits = ((q[half:half + 16] >> shift) & 3).astype(np.int8)
                    sub = np.where(hm[half:half + 16] & m, 0, 4)
                    y[n0 + _ * 32 + half: n0 + _ * 32 + half + 16] = dl * (bits - sub)
                shift += 2
                m <<= 1
            qi += 32
        out[i * QK:(i + 1) * QK] = y
    return out


def deq_q6_k(raw, n_blocks):
    """ql[128] qh[64] scales[16, int8] d(f16); 210 байт/блок."""
    out = np.empty(n_blocks * QK, dtype=np.float32)
    for i in range(n_blocks):
        b = raw[i * 210:(i + 1) * 210]
        ql, qh, sc, d = b[:128], b[128:192], b[192:208], b[208:210]
        dd = np.frombuffer(d, dtype=np.float16)[0].astype(np.float32)
        scv = np.frombuffer(sc, dtype=np.int8).astype(np.int32)
        y = np.empty(QK, dtype=np.float32)
        for n0 in (0, 128):
            qlv = np.frombuffer(ql[n0 // 2:n0 // 2 + 64], dtype=np.uint8).astype(np.int32)
            qhv = np.frombuffer(qh[n0 // 4:n0 // 4 + 32], dtype=np.uint8).astype(np.int32)
            l = np.arange(32)
            isc = l // 16
            q1 = ((qlv[l] & 0xF) | (((qhv[l] >> 0) & 3) << 4)) - 32
            q2 = ((qlv[l + 32] & 0xF) | (((qhv[l] >> 2) & 3) << 4)) - 32
            q3 = ((qlv[l] >> 4) | (((qhv[l] >> 4) & 3) << 4)) - 32
            q4 = ((qlv[l + 32] >> 4) | (((qhv[l] >> 6) & 3) << 4)) - 32
            y[n0 + l] = dd * scv[isc + 0] * q1
            y[n0 + l + 32] = dd * scv[isc + 2] * q2
            y[n0 + l + 64] = dd * scv[isc + 4] * q3
            y[n0 + l + 96] = dd * scv[isc + 6] * q4
        out[i * QK:(i + 1) * QK] = y
    return out


def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4))


def deq_q4_k(raw, n_blocks):
    """d(f16) dmin(f16) scales[12] qs[128]."""
    out = np.empty(n_blocks * QK, dtype=np.float32)
    for i in range(n_blocks):
        b = raw[i * 144:(i + 1) * 144]
        d, dmin, scales, q = b[:2], b[2:4], b[4:16], b[16:144]
        dd = np.frombuffer(d, dtype=np.float16)[0].astype(np.float32)
        mn = np.frombuffer(dmin, dtype=np.float16)[0].astype(np.float32)
        y = np.empty(QK, dtype=np.float32)
        is_ = 0
        for j0 in range(0, QK, 64):
            sc1, m1 = get_scale_min_k4(is_ + 0, scales)
            sc2, m2 = get_scale_min_k4(is_ + 1, scales)
            d1, mm1 = dd * sc1, mn * m1
            d2, mm2 = dd * sc2, mn * m2
            blk = np.frombuffer(q[j0 // 2: j0 // 2 + 32], dtype=np.uint8)
            y[j0:j0 + 32] = d1 * (blk & 0xF).astype(np.float32) - mm1
            y[j0 + 32:j0 + 64] = d2 * (blk >> 4).astype(np.float32) - mm2
            is_ += 2
        out[i * QK:(i + 1) * QK] = y
    return out


def read_gguf_tensor(blob, name):
    f = blob
    f.seek(0)
    assert f.read(4) == b"GGUF"
    _, nten, nkv = struct.unpack("<IQQ", f.read(20))

    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n)

    def skip(t):
        if t == 8:
            rstr()
        elif t == 9:
            (et,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                skip(et)
        else:
            f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                    10: 8, 11: 8, 12: 8}[t])

    for _ in range(nkv):
        rstr()
        (t,) = struct.unpack("<I", f.read(4))
        skip(t)
    pos_raw = f.tell()
    pos_al = pos_raw + (32 - pos_raw % 32) % 32

    def probe(p):
        f.seek(p)
        hdr = f.read(8)
        if len(hdr) < 8:
            return False
        (n,) = struct.unpack("<Q", hdr)
        if n > 64:
            return False
        s = f.read(n)
        return s and all(32 <= c < 127 for c in s)

    f.seek(pos_al if probe(pos_al) else pos_raw)
    info = {}
    for _ in range(nten):
        nm = rstr().decode("utf-8", "replace")
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (tt,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        info[nm] = (tt, dims, off)
    end_infos = f.tell()
    data_start = end_infos + ((32 - end_infos % 32) % 32)
    return info, data_start


def main():
    src, dst = sys.argv[1], sys.argv[2]
    blob = open(src, "rb")
    info, data_start = read_gguf_tensor(blob, "")
    out = {}
    # диспетчеризация по ИСТИННОМУ ggml id: 11=Q3_K, 12=Q4_K, 14=Q6_K
    DISPATCH = {12: (144, deq_q4_k, "Q4_K"), 14: (210, deq_q6_k, "Q6_K"),
                11: (110, deq_q3_k, "Q3_K")}
    for name in ("token_embd.weight", "output.weight"):
        tt, dims, off = info[name]
        rows, cols = dims[1], dims[0]
        n_blocks = cols // QK
        blockbytes, deq, qname = DISPATCH[tt]
        print(f"{name}: тип id={tt} ({qname}), {rows}x{cols}", flush=True)
        mat = np.empty((rows, cols), dtype=np.float16)
        blob.seek(data_start + off)
        data = blob.read(rows * n_blocks * blockbytes)
        for r in range(rows):
            row = deq(data[r * n_blocks * blockbytes:(r + 1) * n_blocks * blockbytes], n_blocks)
            mat[r] = row.astype(np.float16)
        out[name] = mat
    np.savez_compressed(dst, **out)
    print(f"saved {dst}: " + ", ".join(f"{k} {v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    main()
