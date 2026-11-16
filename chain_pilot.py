"""Авто-цепочка пилота: дождаться сборки -> дамп -> тренировка -> экспорт.

Каждая фаза логируется, статус в chain_status.txt; сбой любой фазы
останавливает цепочку с диагнозом. Запуск: python chain_pilot.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "llama.cpp-src", "build_local", "bin")
LOCAL = r"F:\Projects\lm-studio-serv\llama-b12345-for-draft-train\local_v2"
CORPUS = os.path.join(HERE, "corpus")
BUILD_LOG = os.path.join(HERE, "_tools", "build.log")
STATUS = os.path.join(HERE, "chain_status.txt")
VENV_PY = os.path.join(HERE, "train", "venv", "Scripts", "python.exe")
MODEL = r"F:\Data\Models\LLM\Occamy-MiniPlus-V2.1\Occamy-1.0.APEX-I-MiniPlus-V2.1.gguf"


def status(msg):
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print("[chain] " + msg, flush=True)


def run(cmd, cwd=None, log=None, timeout=10 ** 5):
    with open(log, "ab") if log else open(os.devnull, "wb") as sink:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace")[-2000:]


# 1) ждать билд
status("фаза1: ждём сборку (llama-imatrix.exe в build_local/bin)")
exe = os.path.join(BIN, "llama-imatrix.exe")
t0 = time.time()
while not os.path.exists(exe):
    if time.time() - t0 > 3600:
        status("фаза1 ПРОВАЛ: сборка не завершилась за час")
        sys.exit(1)
    time.sleep(30)
status("фаза1 ок: сборка завершена")

# 2) свежий бинарь + дамп
subprocess.run(["cmd", "/c", f"copy /y \"{exe}\" \"{LOCAL}\\llama-imatrix.exe\""],
               capture_output=True)
status("фаза2: redump 340 чанков")
dump = os.path.join(CORPUS, "train_dump.bin")
if os.path.exists(dump):
    os.remove(dump)
rc, out = run([
    os.path.join(LOCAL, "llama-imatrix.exe"),
    "-m", MODEL,
    "-f", os.path.join(CORPUS, "corpus_train.txt"),
    "-c", "2048", "-b", "2048", "-ub", "2048", "-ngl", "99",
    "--n-cpu-moe", "2", "-t", "8", "--chunks", "340", "--no-ppl",
    "--parse-special",
    "--dump-layer-inp", "2,7,12,17,23,28,33,38",
    "--dump-proj", os.path.join(CORPUS, "proj.dfpj"),
    "--dump-topk", "16",
    "--dump-file", dump,
], log=os.path.join(CORPUS, "redump.log"))
if rc != 0 or not os.path.exists(dump) or os.path.getsize(dump) < 10 ** 9:
    status(f"фаза2 ПРОВАЛ rc={rc}: {out[-300:]}")
    sys.exit(2)
status(f"фаза2 ок: дамп {os.path.getsize(dump)/1e9:.1f} ГБ")

# 3) тренировка
status("фаза3: тренировка 1200 шагов")
rc, out = run([
    VENV_PY, os.path.join(HERE, "train_draft.py"), dump,
    "--steps", "1200", "--ctx", "256", "--bs", "6",
], log=os.path.join(HERE, "train", "train.log"))
if rc != 0:
    status(f"фаза3 ПРОВАЛ rc={rc}: {out[-300:]}")
    sys.exit(3)
status("фаза3 ок: чекпоинт написан")

# 4) экспорт
status("фаза4: экспорт BF16")
tmpl = r"F:\Data\Models\LLM\Qwen36-DFlash2-draft\Qwen3.6-35B-A3B-DFlash2-Q4_K_M-tier1.gguf"
bf16 = os.path.join(CORPUS, "ru_draft-BF16.gguf")
rc, out = run([VENV_PY, os.path.join(HERE, "export_gguf.py"), tmpl, bf16],
              log=os.path.join(CORPUS, "export.log"))
if rc != 0 or not os.path.exists(bf16):
    status(f"фаза4 ПРОВАЛ rc={rc}: {out[-300:]}")
    sys.exit(4)
status(f"фаза4 ок: {bf16} - ПИЛОТ ГОТОВ ДЛЯ КВАНТА И ГЕЙТА")
