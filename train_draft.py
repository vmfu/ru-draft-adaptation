"""R3: дообучение DFlash2-драфта на RU-корпусе (учитель = боевой Occamy).

Вход: DFSH-дамп (projected feats via --dump-proj; fc/hidden_norm заморожены
- вендоренный forward без повторной проекции, математически тождественно).

Цель: KL/soft-CE к target top-k распределению на блоках 16 токенов,
контекст - спроецированные фичи ствола. Тренируем слои/conv; селектор и
словарные компоненты заморожены (пилот, путь jzinno).

python train_draft.py <dump.bin> [--steps 1200] [--ctx 256] [--bs 6]
Сторожа: шаг >300с или NaN -> чекпоинт и стоп; лог каждые 10 шагов.
"""
import argparse
import array
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "train", "base")
TABLES = os.path.join(HERE, "corpus", "target_tables.npz")
OUTDIR = os.path.join(HERE, "train", "out")
os.makedirs(OUTDIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--bs", type=int, default=6)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=60)
ap.add_argument("--block", type=int, default=16)
args = ap.parse_args()

torch.manual_seed(2026)

print("[train] грузим модель...", flush=True)
from dflash.model import DFlash2DraftModel  # noqa: E402

model = DFlash2DraftModel.from_pretrained(BASE, dtype=torch.bfloat16)
cfg = model.config
MASK_ID = int(cfg.dflash_config["mask_token_id"])
VOCAB = int(cfg.vocab_size)
HID = int(cfg.hidden_size)

tables = np.load(TABLES)
emb_np = tables["token_embd.weight"].astype(np.float32)
head_np = tables["output.weight"].astype(np.float32)
embd = torch.nn.Embedding.from_pretrained(torch.from_numpy(emb_np), freeze=True).cuda().to(torch.bfloat16)
lm_head = torch.nn.Linear(HID, VOCAB, bias=False)
lm_head.weight = torch.nn.Parameter(torch.from_numpy(head_np).cuda().to(torch.bfloat16), requires_grad=False)
lm_head.eval()
print(f"[train] модель ok: mask_id={MASK_ID} vocab={VOCAB}", flush=True)

# --- заморозка: fc/hidden_norm/selector; тренируем слои+conv+norm ---
for n, p in model.named_parameters():
    p.requires_grad = n.startswith("layers.") or n == "norm.weight"
model = model.cuda().to(torch.bfloat16)
model.train()

# --- вендоренный forward без fc-проекции (feats уже спроецированы) ---
def forward_projected(m, position_ids, noise_embedding, target_hidden):
    hidden_states = noise_embedding
    position_embeddings = m.rotary_emb(hidden_states, position_ids)
    for layer in m.layers:
        hidden_states = layer(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    return m.norm(hidden_states)


# --- датасет из DFSH ---
print("[train] читаем дамп...", flush=True)
f = open(args.dump, "rb")
assert f.read(4) == b"DFSH"
ver, n_layers = struct.unpack("<II", f.read(8))
layer_ids = struct.unpack("<%dI" % n_layers, f.read(4 * n_layers))
n_embd, proj_dim, topk, n_vocab = struct.unpack("<IIII", f.read(16))
assert proj_dim == HID, f"ожидали projected-дамп ({HID}), получили proj={proj_dim}"
assert n_vocab == VOCAB
feat_dim = proj_dim
records = []
while True:
    hdr = f.read(4)
    if len(hdr) < 4:
        break
    (n_tok,) = struct.unpack("<I", hdr)
    toks = np.frombuffer(f.read(4 * n_tok), dtype=np.int32)
    raw = f.read(4 * (topk * 2 + feat_dim) * n_tok)
    ids_all = np.frombuffer(raw, dtype=np.int32).reshape(n_tok, topk * 2 + feat_dim)[:, :topk]
    body = np.frombuffer(raw, dtype=np.float32).reshape(n_tok, topk * 2 + feat_dim)
    tk_ids = ids_all.astype(np.int64)
    tk_lg = body[:, topk:topk * 2].copy()
    feats = body[:, topk * 2:].copy()
    records.append((toks, tk_ids, tk_lg, feats))
f.close()
total = sum(len(r[0]) for r in records)
print(f"[train] записей {len(records)}, токенов {total}", flush=True)
# fail-fast на битый дамп: у 2/3 позиций top-1 id должен быть >0 и логит конечен
chk_i = 0
chk_l = 0.0
n_chk = 0
for r in records[:8]:
    ids, lgs = r[1], r[2]
    chk_i += int((ids[:, 0] > 0).sum())
    chk_l += float(abs(lgs[:, 0]).sum())
    n_chk += len(ids)
assert chk_i > n_chk * 0.5, f"Дамп подозрителен: top-1 id >0 лишь в {chk_i}/{n_chk}"
assert chk_l == chk_l and chk_l > 0, "Дамп подозрителен: top-1 логиты нулевые"
print(f"[train] цели валидны (top1>0: {chk_i}/{n_chk})", flush=True)

# в GPU-память - только фичи и топ-k (токены маленькие)
BIG = [torch.from_numpy(r[3]).cuda() for r in records]
TKI = [torch.from_numpy(r[1]).cuda() for r in records]
TKL = [torch.from_numpy(r[2]).cuda() for r in records]
TOK = [torch.from_numpy(r[0].astype(np.int64)).cuda() for r in records]

opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min((s + 1) / args.warmup, 0.5 * (1 + np.cos(np.pi * min(s / args.steps, 1.0)))))

BL = args.block
CTX = args.ctx
len_r = np.array([len(r[0]) for r in records], dtype=np.float64)
p_rec = len_r / len_r.sum()

t0 = time.time()
loss_hist = []
for step in range(args.steps):
    ts = time.time()
    if ts - t0 > 300 and step > 0:
        print(f"[train] СТОП-сторож: шаг {step} длится {(ts - t0) / 60:.1f} мин", flush=True)
        break
    t0 = ts
    try:
        ri = np.random.choice(len(records), size=args.bs, p=p_rec)
        noises, tgts, poss, tids, tlgs = [], [], [], [], []
        for r in ri:
            toks, tk_ids, tk_lg, feats = records[r]
            L = len(toks)
            b0 = np.random.randint(CTX, L - BL - 1)
            c0 = b0 - CTX
            pos = torch.arange(c0, b0 + BL, device="cuda")
            blk_tok = TOK[r][b0:b0 + BL]
            if np.random.rand() < 0.5:
                noise = embd(blk_tok)
            else:
                keep = torch.rand(BL, device="cuda") < 0.5
                noise = torch.where(keep.unsqueeze(-1), embd(blk_tok),
                                    embd(torch.full_like(blk_tok, MASK_ID)))
            noises.append(noise)
            tgts.append(BIG[r][c0:b0])
            poss.append(pos)
            tids.append(TKI[r][b0:b0 + BL])
            tlgs.append(TKL[r][b0:b0 + BL])
        noise = torch.stack(noises)
        tgt = torch.stack(tgts).to(torch.bfloat16)
        pos = torch.stack(poss)
        tids = torch.stack(tids).long()
        tlgs = torch.stack(tlgs)

        hidden = forward_projected(model, pos, noise, tgt)
        logits = model.compute_logits(hidden, lm_head).float()
        logq = F.log_softmax(logits, dim=-1)
        p = F.softmax(tlgs.float(), dim=-1)
        loss = -(p * logq.gather(-1, tids)).sum(-1).mean()
        if not torch.isfinite(loss):
            print(f"[train] СТОП: NaN на шаге {step}", flush=True)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        loss_hist.append(float(loss))
        if (step + 1) % 10 == 0:
            m = sum(loss_hist[-50:]) / len(loss_hist[-50:])
            print(f"[train] {step+1}/{args.steps} loss={float(loss):.4f} "
                  f"MA50={m:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{time.time()-ts:.1f}с", flush=True)
        if (step + 1) % 300 == 0:
            torch.save({k: v for k, v in model.state_dict().items()},
                       os.path.join(OUTDIR, "ru_draft.safetensors.tmp.pt"))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print(f"[train] OOM на шаге {step}, пропускаю", flush=True)
            continue
        raise

# --- финал: только обученные тензоры ---
trained = {k: v.cpu() for k, v in model.state_dict().items()
           if any(s in k for s in (".layers.", ".norm.weight"))}
torch.save(trained, os.path.join(OUTDIR, "ru_draft_layers.pt"))
print(f"[train] ГОТОВО: {len(trained)} тензоров -> {OUTDIR}/ru_draft_layers.pt",
      flush=True)
with open(os.path.join(OUTDIR, "train_meta.json"), "w", encoding="utf-8") as mf:
    json.dump({"steps": len(loss_hist), "loss_first": loss_hist[:5],
               "loss_last": loss_hist[-5:],
               "args": vars(args)}, mf, ensure_ascii=False, indent=1)
