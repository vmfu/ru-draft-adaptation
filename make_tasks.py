"""R2.1: генератор задач для симуляции RU-около-кодового корпуса.

Чанкит исходники (наш репо + requests), навешивает RU/EN задания по
шаблонам. Выход: tasks.jsonl {id, component, prompt}.

Микс-цель (Clover-эталон, RU-доля повышенная):
  ruc (RU-код-рассуждение) 40% | rut (RU-текст/терминология) 25%
  enc (EN-код) 15% | tools 10% | reasoning 10%

python make_tasks.py [цель_токенов_выхлопа]   (по умолчанию 1250000:
при среднем выхлопе ~700 ток/задача ~ <=1900 задач)
"""
import glob
import json
import os
import random
import sys

random.seed(20261114)

REPO_SELF = "F:/Projects/lm-studio-serv"
REPO_EXT = "F:/Projects/lm-studio-serv/_ru_draft/reposrc/requests"
OUT = os.path.join(os.path.dirname(__file__), "tasks.jsonl")

CHUNK_LINES = 70  # ~ 400-900 токенов кода на чанк

RUC = [
    "Расставь логические коммиты по этому коду: разбей изменения на осмысленные шаги и объясни каждый по-русски одним-двумя предложениями:\n\n```\n{chunk}\n```",
    "Объясни по-русски, что делает этот фрагмент, и где он может сломаться:\n\n```\n{chunk}\n```",
    "Дай code review этого кода по-русски: найди проблемы, оцени качество по 10-балльной шкале, предложи исправления:\n\n```\n{chunk}\n```",
    "Переведи терминологию и комментарии в этом коде на русский, сохраняя стиль:\n\n```\n{chunk}\n```",
    "Найди баг или потенциальную проблему в этом фрагменте и объясни по-русски, как её воспроизвести:\n\n```\n{chunk}\n```",
    "Предложи рефакторинг этого фрагмента по-русски: что упростить, какие паттерны применить, и напиши улучшенную версию:\n\n```\n{chunk}\n```",
    "Напиши по-русски модульные тесты для этого кода и объясни, что покрывает каждый:\n\n```\n{chunk}\n```",
]
RUT = [
    "Объясни по-русски простыми словами, о чём этот текст, и составь глоссарий терминов:\n\n{chunk}",
    "Перескажи по-русски этот фрагмент документации и укажи, какие вопросы он оставляет открытыми:\n\n{chunk}",
    "Составь по-русски пять вопросов для проверки понимания этого текста и ответы на них:\n\n{chunk}",
]
ENC = [
    "Annotate this code into logical commits and explain each briefly:\n\n```\n{chunk}\n```",
    "Review this code: list issues and rate quality out of 10:\n\n```\n{chunk}\n```",
    "Explain what this code does and where it can fail:\n\n```\n{chunk}\n```",
]
TOOLS = [
    "You have a read_file tool. Read the file config.py in this project and summarize it in Russian. Code context:\n\n```\n{chunk}\n```",
    "You have a bash tool. Describe the exact commands you would run to test this code, then explain them in Russian:\n\n```\n{chunk}\n```",
    "You have an edit tool. Show the precise edit you would make to fix the worst issue here, then explain it in Russian:\n\n```\n{chunk}\n```",
]
REASON = [
    "Пусть этот код обрабатывает поток данных. Оцени по-русски его асимптотику и узкие места, рассуждая шаг за шагом:\n\n```\n{chunk}\n```",
    "Придумай по-русски три граничных случая, на которых этот код упадёт, и мысленно прогони каждый:\n\n```\n{chunk}\n```",
]

PY = ("*.py",)
MD = ("*.md", "*.rst")


def chunks_of(path):
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return
    for i in range(0, len(lines), CHUNK_LINES):
        piece = "\n".join(lines[i:i + CHUNK_LINES]).strip()
        if 200 < len(piece) < 6000:
            yield piece


def collect(root, patterns, skip_dirs=(".git", "__pycache__", "_ru_draft", "llama.cpp-src")):
    out = []
    for pat in patterns:
        for path in glob.glob(os.path.join(root, "**", pat), recursive=True):
            if any(sd in path for sd in skip_dirs):
                continue
            out.append(path)
    return out


code_chunks = []
for repo in (REPO_SELF, REPO_EXT):
    for path in collect(repo, PY):
        for piece in chunks_of(path):
            code_chunks.append((path, piece))
text_chunks = []
for repo in (REPO_SELF, REPO_EXT):
    for path in collect(repo, MD):
        for piece in chunks_of(path):
            text_chunks.append((path, piece))

print(f"чанков: код {len(code_chunks)}, текст {len(text_chunks)}")

target_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 1_250_000
AVG_OUT = 700
n_total = min(target_tokens // AVG_OUT, 4000)

mix = {"ruc": 0.40, "rut": 0.25, "enc": 0.15, "tools": 0.10, "reasoning": 0.10}
counts = {k: int(n_total * v) for k, v in mix.items()}
print(f"задач: {counts} (сумма {sum(counts.values())})")

tasks = []
iid = 0
for comp, n in counts.items():
    pool = code_chunks if comp != "rut" else text_chunks
    tmpl = {"ruc": RUC, "rut": RUT, "enc": ENC, "tools": TOOLS, "reasoning": REASON}[comp]
    for _ in range(n):
        path, piece = random.choice(pool)
        t = random.choice(tmpl)
        tasks.append({"id": iid, "component": comp, "source": path,
                      "prompt": t.format(chunk=piece)})
        iid += 1

random.shuffle(tasks)
with open(OUT, "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"готово: {OUT} ({len(tasks)} задач)")
