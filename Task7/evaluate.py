import sys
import csv
import time
from pathlib import Path
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
TASK6_DIR = CURRENT_DIR.parent / "Task5"
sys.path.insert(0, str(TASK6_DIR))

from rag_bot_secure import KopruluRAGBotSecure, SecurityConfig

GOLDEN_FILE = CURRENT_DIR / "golden_questions.txt"
LOG_FILE = CURRENT_DIR / "logs.csv"
REPORT_FILE = CURRENT_DIR / "evaluation_report.md"

NOT_ANSWERED_MARKERS = [
    "не знаю", "нет такой информации", "не содержит",
    "не упоминается", "отсутствует", "нет данных",
    "не могу ответить", "не нашел", "не нашёл",
    "нет информации", "не располагаю", "в моей базе",
    "в контексте нет", "недостаточно информации",
    "i don't know", "no information",
]


def load_golden():
    questions = []
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("ID|"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue
            keywords = [k.strip() for k in parts[3].split(",") if k.strip()]
            questions.append({
                "id": int(parts[0]),
                "question": parts[1],
                "expected_status": parts[2],
                "keywords": keywords,
                "topic": parts[4],
            })
    return questions


def classify(answer, expected_status, keywords):
    answer_lower = answer.lower()

    is_refusal = any(m in answer_lower for m in NOT_ANSWERED_MARKERS)
    if len(answer.strip()) < 30:
        is_refusal = True

    actual = "not_answered" if is_refusal else "answered"

    found = [k for k in keywords if k.lower() in answer_lower]
    missing = [k for k in keywords if k.lower() not in answer_lower]

    if expected_status == "answered":
        if keywords:
            ratio = len(found) / len(keywords)
            status_ok = 1.0 if actual == "answered" else 0.0
            confidence = ratio * 0.7 + status_ok * 0.3
        else:
            confidence = 1.0 if actual == "answered" else 0.0
        correct = (actual == "answered") and (len(found) > 0 or not keywords)
    else:
        confidence = 1.0 if actual == "not_answered" else 0.0
        correct = (actual == "not_answered")

    return {
        "actual_status": actual,
        "correct": correct,
        "confidence": round(confidence, 2),
        "keywords_found": found,
        "keywords_missing": missing,
    }


def get_sources(bot, question):
    try:
        docs = bot.retriever.invoke(question)
        sources = list(set(d.metadata.get("source", "?") for d in docs))
        return len(docs), "; ".join(sources)
    except Exception:
        return 0, ""


def write_csv_header():
    with open(LOG_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "id", "question", "topic",
            "expected_status", "actual_status", "correct",
            "confidence", "answer_length", "chunks_found",
            "sources", "keywords_found", "keywords_missing",
            "elapsed_sec", "answer_preview",
        ])


def write_csv_row(row):
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def generate_report(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0

    lines = [
        "# Отчет об оценке RAG-бота",
        "",
        f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Общая статистика",
        "",
        "| Метрика | Значение |",
        "|---------|----------|",
        f"| Всего вопросов | {total} |",
        f"| Корректных | {correct} |",
        f"| Точность | {accuracy:.1%} |",
        "",
        "## Результаты по каждому вопросу",
        "",
    ]

    for r in results:
        mark = "PASS" if r["correct"] else "FAIL"
        lines.extend([
            f"### [{mark}] Вопрос {r['id']}: {r['question']}",
            "",
            f"- Тема: {r['topic']}",
            f"- Ожидалось: {r['expected_status']}, получено: {r['actual_status']}",
            f"- Уверенность: {r['confidence']}",
            f"- Ключевые слова найдены: {r['keywords_found']}",
            f"- Ключевые слова пропущены: {r['keywords_missing']}",
            f"- Источники: {r['sources']}",
            f"- Длина ответа: {r['answer_length']}",
            f"- Ответ: {r['answer_preview']}",
            "",
        ])

    # Анализ по темам
    topics = {}
    for r in results:
        t = r["topic"]
        if t not in topics:
            topics[t] = {"total": 0, "correct": 0, "failed": []}
        topics[t]["total"] += 1
        if r["correct"]:
            topics[t]["correct"] += 1
        else:
            topics[t]["failed"].append(r["question"])

    lines.extend([
        "## Анализ по темам",
        "",
        "| Тема | Всего | Корректно | Статус |",
        "|------|-------|-----------|--------|",
    ])

    for t, d in sorted(topics.items()):
        status = "OK" if d["correct"] == d["total"] else "ПРОБЛЕМА"
        lines.append(f"| {t} | {d['total']} | {d['correct']} | {status} |")

    # Выявленные пробелы
    weak = {t: d for t, d in topics.items() if d["correct"] < d["total"]}

    lines.extend(["", "## Выявленные пробелы", ""])

    if weak:
        lines.append(f"Обнаружено проблемных тем: {len(weak)}")
        lines.append("")
        for t, d in weak.items():
            lines.append(f"- **{t}** ({d['correct']}/{d['total']} корректных):")
            for q in d["failed"]:
                lines.append(f"  - \"{q}\"")
    else:
        lines.append("Пробелов не выявлено.")

    # Рекомендации -- формируются динамически
    lines.extend(["", "## Рекомендации по улучшению базы знаний", ""])

    if not weak:
        lines.append("База знаний полностью покрывает золотой набор вопросов.")
    else:
        rec_num = 1

        # Темы, где бот не ответил хотя должен был
        missed_answers = [
            r for r in results
            if r["expected_status"] == "answered" and not r["correct"]
        ]
        if missed_answers:
            missed_topics = set(r["topic"] for r in missed_answers)
            lines.append(f"{rec_num}. Добавить или расширить документы по темам, "
                         f"где бот не смог ответить:")
            for t in sorted(missed_topics):
                questions = [r["question"] for r in missed_answers if r["topic"] == t]
                lines.append(f"   - {t}:")
                for q in questions:
                    lines.append(f"     - \"{q}\"")
            rec_num += 1

        # Темы, где бот ответил хотя не должен был
        false_answers = [
            r for r in results
            if r["expected_status"] != "answered" and r["actual_status"] == "answered"
        ]
        if false_answers:
            lines.append(f"{rec_num}. Проверить корректность отказа бота "
                         f"по следующим вопросам (бот ответил, хотя информация "
                         f"отсутствует в базе):")
            for r in false_answers:
                lines.append(f"   - \"{r['question']}\" (тема: {r['topic']})")
            rec_num += 1

        # Пропущенные ключевые слова
        missing_kw = [
            r for r in results
            if r["keywords_missing"] and r["expected_status"] == "answered"
        ]
        if missing_kw:
            lines.append(f"{rec_num}. Улучшить полноту ответов "
                         f"(бот отвечает, но пропускает важные детали):")
            for r in missing_kw:
                lines.append(f"   - \"{r['question']}\": "
                             f"пропущены слова {r['keywords_missing']}")
            rec_num += 1

        # Общая рекомендация
        lines.append(f"{rec_num}. После внесения изменений повторно запустить "
                     f"evaluate.py для проверки улучшений.")

    lines.append("")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("АВТОМАТИЧЕСКАЯ ОЦЕНКА RAG-БОТА")
    print("=" * 60)

    questions = load_golden()
    print(f"Загружено вопросов: {len(questions)}")

    config = SecurityConfig(
        enable_preprompt=True,
        enable_chunk_filter=True,
        enable_output_filter=True,
    )
    bot = KopruluRAGBotSecure(security_config=config)

    write_csv_header()
    results = []

    for q in questions:
        print(f"\n[{q['id']}/{len(questions)}] {q['question']}")

        start = time.time()
        bot_result = bot.ask(q["question"])
        elapsed = round(time.time() - start, 2)

        answer = bot_result["answer"]
        chunks_found, sources = get_sources(bot, q["question"])

        ev = classify(answer, q["expected_status"], q["keywords"])

        mark = "PASS" if ev["correct"] else "FAIL"
        print(f"  [{mark}] {ev['actual_status']} "
              f"(ожидалось: {q['expected_status']}), "
              f"confidence={ev['confidence']}")

        preview = answer[:200].replace("\n", " ").replace(",", ";")

        write_csv_row([
            datetime.now().isoformat(),
            q["id"],
            q["question"],
            q["topic"],
            q["expected_status"],
            ev["actual_status"],
            ev["correct"],
            ev["confidence"],
            len(answer),
            chunks_found,
            sources,
            "; ".join(ev["keywords_found"]),
            "; ".join(ev["keywords_missing"]),
            elapsed,
            preview,
        ])

        results.append({
            "id": q["id"],
            "question": q["question"],
            "topic": q["topic"],
            "expected_status": q["expected_status"],
            "actual_status": ev["actual_status"],
            "correct": ev["correct"],
            "confidence": ev["confidence"],
            "keywords_found": ev["keywords_found"],
            "keywords_missing": ev["keywords_missing"],
            "sources": sources,
            "answer_length": len(answer),
            "answer_preview": preview,
        })

    report = generate_report(results)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"\nТочность: {correct}/{total} ({correct/total:.1%})")
    print(f"Лог: {LOG_FILE}")
    print(f"Отчет: {REPORT_FILE}")


if __name__ == "__main__":
    main()