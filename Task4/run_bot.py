from rag_bot import KopruluRAGBot


def main():
    try:
        bot = KopruluRAGBot()
    except Exception as e:
        print(f"[FATAL] Ошибка инициализации: {e}")        
        return

    print("\n" + "=" * 80)
    print("RAG-бот по копрулуским легенда")
    print("Для выхода: exit / e")
    print("=" * 80 + "\n")

    while True:
        query = input("Ваш вопрос: ").strip()
        if query.lower() in ["exit", "e"]:
            print("До встречи в секторе Копрулу!")
            break

        if not query:
            continue

        print("\nРазмышляю...\n")
        print(bot.ask(query))


if __name__ == "__main__":
    main()