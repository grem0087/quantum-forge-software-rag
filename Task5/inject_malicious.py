import sys
from pathlib import Path

TASK3_DIR = Path(__file__).resolve().parent.parent / "Task3"
sys.path.append(str(TASK3_DIR))

from config import CHROMA_PATH as RELATIVE_CHROMA_PATH, EMBEDDING_MODEL_NAME, COLLECTION_NAME

import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def inject_malicious_document():
    chroma_path = str(TASK3_DIR / RELATIVE_CHROMA_PATH)

    # Тот же эмбеддер, что и в основном пайплайне
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="",
    )

    # Читаем вредоносный файл
    malicious_path = Path(__file__).parent / "malicious_document.txt"
    malicious_text = malicious_path.read_text(encoding="utf-8").strip()

    print(f"[INJECT] Вредоносный текст: {malicious_text!r}")

    # Получаем эмбеддинг
    embedding = embedding_function.embed_documents([malicious_text])[0]

    # Подключаемся к ChromaDB и добавляем документ
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=COLLECTION_NAME)

    doc_id = "malicious_injection_001"

    # Удаляем старый, если есть (для повторных запусков)
    try:
        collection.delete(ids=[doc_id])
    except Exception:
        pass

    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[malicious_text],
        metadatas=[{"source": "malicious_document.txt", "type": "injection"}],
    )

    print(f"[INJECT] Документ добавлен в коллекцию '{COLLECTION_NAME}'")
    print(f"[INJECT] Всего документов: {collection.count()}")


if __name__ == "__main__":
    inject_malicious_document()