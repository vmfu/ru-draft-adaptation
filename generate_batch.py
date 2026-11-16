"""R2-батчевый: генерация корпуса пачками с рестартом колеи между ними.

Доктрина оператора: llama-сервер деградирует по мере заполнения кеша
(96 t/s на 170-й задаче против 113 на первой) - рестартуем колею
каждые BATCH задач и следим за скоростью каждого запроса: два подряд
медленных (< SPEED_FLOOR) - внеплановый рестарт. Prefill виден в gen_s.

python generate_batch.py [всего_задач] [батч]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "corpus", "tasks.jsonl")
OUT = os.path.join(HERE, "corpus", "generation.jsonl")
BASE = os.path.dirname(HERE)
SRV = os.path.join(BASE, "srv.py")
LLAMA_DIR = os.path.join(BASE, "llama-b12345-for-draft-train", "patched")

BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 150
TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
MAX_TOKENS = 3072
TEMP = 0.8
SPEED_FLOOR = 40.0     # t/s; ниже - сигнал деградации
STALL_S = 900.0
STALL_N = 3
URL = "http://127.0.0.1:1234/v1/chat/completions"


def health(timeout=360):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if urllib.request.urlopen("http://127.0.0.1:1234/health",
                                      timeout=3).status == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def lane(op):
    env = dict(os.environ, SRV_LLAMA_DIR=LLAMA_DIR)
    subprocess.run([sys.executable, SRV, op], env=env,
                   capture_output=True, timeout=600)
    if op == "stop":
        time.sleep(4)
        return True
    return health()


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
    m = d["choices"][0]["message"]
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return m.get("content") or "", m.get("reasoning_content") or "", ct, time.time() - t0


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
    todo = [t for t in tasks if t["id"] not in done][:TOTAL]
    print(f"[bat] осталось {len(todo)}, батч {BATCH}, пол {SPEED_FLOOR} t/s",
          flush=True)
    lane("stop")
    n_done = 0
    slow_streak = 0
    with open(OUT, "a", encoding="utf-8") as out:
        for i, t in enumerate(todo):
            if n_done >= BATCH and n_done % BATCH == 0:
                print(f"[bat] батч {n_done} - рестарт колеи", flush=True)
                lane("stop")
                if not lane("start"):
                    print("[bat] СТОП: колея не встала", flush=True)
                    return 2
                slow_streak = 0
            try:
                if not health(30):
                    if not lane("start"):
                        return 2
                content, reasoning, ct, dt = post(t["prompt"])
                speed = ct / dt if dt > 0 else 0
                slow_streak = slow_streak + 1 if speed < SPEED_FLOOR else 0
                if slow_streak >= 2:
                    print(f"[bat] деградация ({speed:.0f} t/s) - рестарт",
                          flush=True)
                    lane("stop")
                    if not lane("start"):
                        return 2
                    slow_streak = 0
            except Exception as e:
                print(f"[bat] FAIL id={t['id']}: {e}", flush=True)
                lane("stop")
                if not lane("start"):
                    return 2
                continue
            out.write(json.dumps({
                "id": t["id"], "component": t["component"],
                "prompt": t["prompt"], "content": content,
                "reasoning": reasoning, "completion_tokens": ct,
                "gen_s": round(dt, 1), "t_s": round(speed, 1),
            }, ensure_ascii=False) + "\n")
            out.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f"[bat] {n_done}/{len(todo)} | последний "
                      f"{speed:.0f} t/s | {ct} ток {dt:.0f}с", flush=True)
    print(f"[bat] ЗАВЕРШЕНО: +{n_done} задач", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
