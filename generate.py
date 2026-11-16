"""R2.3: генератор корпуса на живой колее (с бойц-драфтом для скорости).

Читает tasks.jsonl, шлёт промпты на :1234 (колея occamy, ctx 8192,
билд llama-b12345-for-draft-train, НЕ-ру драфт подключён - выхлоп чист
по rejection-семплингу), пишет generation.jsonl append-only.

Сторожа (канон «прослушка процесса»):
  - heartbeat каждые N задач в лог и консоль;
  - stall-аборт: если k задач подряд падают/длятся > STALL_S - стоп;
  - resume: готовые id читаются из generation.jsonl, пропускаются;
  - авто-завершение по исчерпании задач или LIMIT.

python generate.py [лимит_задач]
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(__file__)
TASKS = os.path.join(HERE, "corpus", "tasks.jsonl")
OUT = os.path.join(HERE, "corpus", "generation.jsonl")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
STALL_S = 900.0
STALL_N = 3
HEART_EVERY = 10
MAX_TOKENS = 3072
TEMP = 0.8

URL = "http://127.0.0.1:1234/v1/chat/completions"


def post(prompt):
    body = json.dumps({
        "model": "occamy-1.0", "stream": False, "max_tokens": MAX_TOKENS,
        "temperature": TEMP, "top_p": 0.95,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=STALL_S).read())
    msg = d["choices"][0]["message"]
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return msg.get("content") or "", msg.get("reasoning_content") or "", ct, time.time() - t0


def done_ids():
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def main():
    tasks = [json.loads(l) for l in open(TASKS, encoding="utf-8")]
    done = done_ids()
    todo = [t for t in tasks if t["id"] not in done][:LIMIT]
    print(f"[gen] задач всего {len(tasks)}, готово {len(done)}, осталось {len(todo)}",
          flush=True)
    t_start = time.time()
    tokens = 0
    stalls = 0
    with open(OUT, "a", encoding="utf-8") as out:
        for i, t in enumerate(todo):
            try:
                content, reasoning, ct, dt = post(t["prompt"])
                stalls = 0
            except Exception as e:
                stalls += 1
                print(f"[gen] FAIL id={t['id']} ({e}); stalls={stalls}", flush=True)
                if stalls >= STALL_N:
                    print("[gen] СТОП: stall-детектор", flush=True)
                    return 2
                continue
            out.write(json.dumps({
                "id": t["id"], "component": t["component"], "source": t["source"],
                "prompt": t["prompt"], "content": content, "reasoning": reasoning,
                "completion_tokens": ct, "gen_s": round(dt, 1),
            }, ensure_ascii=False) + "\n")
            out.flush()
            tokens += ct
            if (i + 1) % HEART_EVERY == 0:
                rate = tokens / (time.time() - t_start)
                eta_min = (len(todo) - i - 1) * (time.time() - t_start) / (i + 1) / 60
                print(f"[gen] {i+1}/{len(todo)} | {tokens} ток | {rate:.0f} ток/с | "
                      f"ETA {eta_min:.0f} мин", flush=True)
    print(f"[gen] ЗАВЕРШЕНО: {len(todo)} задач, {tokens} токенов, "
          f"{(time.time()-t_start)/60:.1f} мин", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
