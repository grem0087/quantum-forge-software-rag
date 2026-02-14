import os
import sys
import shutil
import logging
import traceback
from pathlib import Path
from datetime import datetime

# Пути
CURRENT_DIR = Path(__file__).resolve().parent
TASK3_DIR = CURRENT_DIR.parent / "Task3"
sys.path.append(str(TASK3_DIR))

DATA_UPLOAD_DIR = CURRENT_DIR / "data_upload"
LOG_FILE = CURRENT_DIR / "update_index_log.txt"

# Параметры индексации (те же, что в rebuild_index.py)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "koprulu_myths"
CHUNK_SIZE = 950
CHUNK_OVERLAP = 120

# Путь к ChromaDB берем из config Task3, либо задаем вручную
try:
    from config import CHROMA_PATH as RELATIVE_CHROMA_PATH
    CHROMA_PATH = TASK3_DIR / RELATIVE_CHROMA_PATH
except ImportError:
    CHROMA_PATH = TASK3_DIR / "chroma_db"


# -------------------------------------------
# Настройка логирования
# -------------------------------------------

def setup_logging():
    """Настраивает логирование в файл и в консоль."""
    logger = logging.getLogger("update_index")
    logger.setLevel(logging.DEBUG)

    # Формат
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Файловый хендлер (дописывает в конец)
    file_handler = logging.FileHandler(
        str(LOG_FILE), mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # Консольный хендлер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# -------------------------------------------
# Основная логика
# -------------------------------------------

def scan_new_files(upload_dir):
    """
    Сканирует папку на наличие новых .md файлов.
    Возвращает список путей.
    """
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)
        return []

    files = sorted(upload_dir.glob("*.md"))
    return files


def process_file(md_path, splitter, embedding_function, collection, logger):
    """
    Обрабатывает один .md файл:
      - читает текст
      - разбивает на чанки
      - генерирует эмбеддинги
      - добавляет в коллекцию ChromaDB
      - удаляет файл после успешной обработки

    Возвращает количество добавленных чанков.
    """
    logger.info(f"Обрабатываю файл: {md_path.name}")

    # Чтение
    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        logger.warning(f"Файл пустой, пропускаю: {md_path.name}")
        os.remove(md_path)
        return 0

    # Разбиение на чанки
    doc_chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": md_path.name, "title": md_path.stem}],
    )

    if not doc_chunks:
        logger.warning(f"Нет чанков после разбиения: {md_path.name}")
        os.remove(md_path)
        return 0

    # Подготовка данных
    ids = []
    texts = []
    metadatas = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, chunk in enumerate(doc_chunks):
        # ID включает timestamp для уникальности при повторных загрузках
        chunk_id = f"{md_path.stem}-{timestamp}-chunk-{i}"
        ids.append(chunk_id)
        texts.append(chunk.page_content)
        metadatas.append({
            "source": md_path.name,
            "title": md_path.stem,
            "chunk_index": i,
            "start_index": chunk.metadata.get("start_index", 0),
            "indexed_at": timestamp,
        })

    # Генерация эмбеддингов
    logger.info(f"  Генерирую эмбеддинги для {len(texts)} чанков...")
    embeddings = embedding_function.embed_documents(texts)

    # Проверка на дубликаты: удаляем старые чанки из того же файла
    existing = collection.get(where={"source": md_path.name})
    if existing and existing["ids"]:
        logger.info(f"  Удаляю {len(existing['ids'])} старых чанков из {md_path.name}")
        collection.delete(ids=existing["ids"])

    # Добавление в коллекцию
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts,
    )

    logger.info(f"  Добавлено {len(doc_chunks)} чанков из {md_path.name}")

    # Удаление обработанного файла
    os.remove(md_path)
    logger.info(f"  Файл удален: {md_path.name}")

    return len(doc_chunks)


def main():
    """Главная функция: полный цикл обновления индекса."""

    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("ЗАПУСК ОБНОВЛЕНИЯ ИНДЕКСА")
    logger.info("=" * 60)

    start_time = datetime.now()
    total_new_chunks = 0
    files_processed = 0
    files_failed = 0
    errors = []

    try:
        # 1. Сканирование
        new_files = scan_new_files(DATA_UPLOAD_DIR)
        logger.info(f"Папка сканирования: {DATA_UPLOAD_DIR}")
        logger.info(f"Найдено новых файлов: {len(new_files)}")

        if not new_files:
            logger.info("Нет новых файлов для обработки.")
            logger.info(f"Завершено за {datetime.now() - start_time}")
            logger.info("")
            return

        # 2. Инициализация компонентов
        logger.info("Инициализация эмбеддера...")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        import chromadb

        embedding_function = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="",
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

        logger.info(f"Подключение к ChromaDB: {CHROMA_PATH}")
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

        index_size_before = collection.count()
        logger.info(f"Размер индекса до обновления: {index_size_before} чанков")

        # 3. Обработка каждого файла
        for md_path in new_files:
            try:
                chunks_added = process_file(
                    md_path, splitter, embedding_function, collection, logger
                )
                total_new_chunks += chunks_added
                files_processed += 1
            except Exception as e:
                files_failed += 1
                error_msg = f"Ошибка при обработке {md_path.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                # Файл НЕ удаляется при ошибке -- останется для повторной попытки

        # 4. Итоги
        index_size_after = collection.count()
        elapsed = datetime.now() - start_time

        logger.info("-" * 40)
        logger.info("ИТОГИ ОБНОВЛЕНИЯ:")
        logger.info(f"  Время запуска:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Затрачено:        {elapsed}")
        logger.info(f"  Файлов обработано: {files_processed}")
        logger.info(f"  Файлов с ошибкой:  {files_failed}")
        logger.info(f"  Новых чанков:      {total_new_chunks}")
        logger.info(f"  Размер индекса:    {index_size_before} -> {index_size_after}")
        logger.info(f"  Ошибки:            {len(errors)}")

        if errors:
            for err in errors:
                logger.info(f"    - {err}")

        # Однострочная сводка (удобно для быстрого просмотра лога)
        summary = (
            f"index updated at {start_time.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"{files_processed} files added, "
            f"{total_new_chunks} chunks, "
            f"index size {index_size_after}, "
            f"{len(errors)} errors"
        )
        logger.info(f"SUMMARY: {summary}")
        logger.info("")

    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        logger.debug(traceback.format_exc())
        logger.info("")


if __name__ == "__main__":
    main()