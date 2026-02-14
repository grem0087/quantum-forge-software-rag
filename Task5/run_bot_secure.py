from rag_bot_secure import KopruluRAGBotSecure, SecurityConfig


def print_result(result: dict):
    """Вывод результата."""
    print("\n" + "─" * 60)
    print(f"Статус: {result['status']}")
    print(f"Чанков найдено: {result['total_chunks_retrieved']}, "
          f"использовано: {result['safe_chunks_used']}")

    if result["blocked_chunks"]:
        print(f"\n Заблокировано чанков: {len(result['blocked_chunks'])}")

    if result["output_filtered"]:
        print(f"\n Ответ отфильтрован: {result['filter_reason']}")

    print(f"\n Ответ:\n{result['answer']}")
    print("─" * 60 + "\n")


def main():
    config = SecurityConfig(
        enable_preprompt=True,
        enable_chunk_filter=True,
        enable_output_filter=True,
    )

    try:
        bot = KopruluRAGBotSecure(security_config=config)
    except Exception as e:
        print(f"[FATAL] Ошибка инициализации: {e}")
        return

    print("=" * 60)
    print(" Защищённый RAG-бот по копрулуским легендам")
    print("=" * 60)
    print("Команды:")
    print("  exit / e         — выход")
    print("  !toggle preprompt  — вкл/выкл pre-prompt защиту")
    print("  !toggle chunks     — вкл/выкл фильтр чанков")
    print("  !toggle output     — вкл/выкл фильтр ответа")
    print("  !toggle all        — вкл/выкл ВСЕ слои")
    print("  !status            — показать состояние защиты")
    print("=" * 60 + "\n")

    while True:
        query = input("Ваш вопрос: ").strip()

        if query.lower() in ["exit", "e"]:
            print("До встречи в секторе Копрулу!")
            break

        if query == "!status":
            print(f"\n  Pre-prompt:    {'ON' if bot.config.enable_preprompt else 'OFF'}")
            print(f"  Chunk filter:  {'ON' if bot.config.enable_chunk_filter else 'OFF'}")
            print(f"  Output filter: {'ON' if bot.config.enable_output_filter else 'OFF'}\n")
            continue

        if query.startswith("!toggle"):
            parts = query.split()
            if len(parts) < 2:
                print("Укажите: preprompt, chunks, output или all")
                continue

            target = parts[1]

            if target == "preprompt":
                bot.config.enable_preprompt = not bot.config.enable_preprompt
                print(f"  Pre-prompt: {'ON' if bot.config.enable_preprompt else 'OFF'}")
            elif target == "chunks":
                bot.config.enable_chunk_filter = not bot.config.enable_chunk_filter
                print(f"  Chunk filter: {'ON' if bot.config.enable_chunk_filter else 'OFF'}")
            elif target == "output":
                bot.config.enable_output_filter = not bot.config.enable_output_filter
                print(f"  Output filter: {'ON' if bot.config.enable_output_filter else 'OFF'}")
            elif target == "all":
                new_state = not bot.config.enable_preprompt
                bot.config.enable_preprompt = new_state
                bot.config.enable_chunk_filter = new_state
                bot.config.enable_output_filter = new_state
                print(f"  Все слои: {'ON' if new_state else 'OFF'}")
            else:
                print("Неизвестный слой. Доступно: preprompt, chunks, output, all")
            continue

        if not query:
            continue

        print("\nРазмышляю...")
        result = bot.ask(query)
        print_result(result)


if __name__ == "__main__":
    main()