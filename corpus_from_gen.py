"""Сборка тренировочного корпуса из generation.jsonl -> corpus_train.txt.

Рендерит каждую задачу через Qwen-шаблон (те же спец-токены, что
--parse-back в дамп-прогоне), склеивает в один поток. Дамп-пасс
прогонит этот файл чанками по 2048 токенов.

python corpus_from_gen.py [макс_задач]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "corpus", "generation.jsonl")
OUT = os.path.join(HERE, "corpus", "corpus_train.txt")

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9

parts = []
n = 0
tokens_est = 0
with open(GEN, encoding="utf-8") as f:
    for line in f:
        if n >= limit:
            break
        try:
            r = json.loads(line)
        except Exception:
            continue
        content = (r.get("content") or "").strip()
        reasoning = (r.get("reasoning") or "").strip()
        if not content and len(reasoning) < 200:
            continue  # совсем пустое - мимо
        think = f"<think>\n{reasoning}\n</think>\n\n" if reasoning else ""
        # обрезанный thinking без ответа тоже годится: это валидный
        # RU-текст учителя (трейн ест любое teacher-forced распределение)
        body = f"{think}{content}" if content else think.rstrip()
        parts.append(
            f"<|im_start|>user\n{r['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{body}<|im_end|>\n"
        )
        tokens_est += r.get("completion_tokens", 0)
        n += 1

text = "\n".join(parts)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text)
mb = len(text.encode("utf-8")) / 1e6
print(f"{n} диалогов, ~{tokens_est} ток. выхлопа, {mb:.1f} МБ -> {OUT}")
