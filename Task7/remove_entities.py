import sys
import json
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
TASK3_DIR = CURRENT_DIR.parent / "Task3"
sys.path.append(str(TASK3_DIR))

from config import CHROMA_PATH as RELATIVE_CHROMA_PATH, COLLECTION_NAME
import chromadb

ENTITIES_TO_REMOVE = {
    "zagara": ["01-zagara-the-ancient-broodmother-chunk-0",
               "01-zagara-the-ancient-broodmother-chunk-1",
               "01-zagara-the-ancient-broodmother-chunk-2"],
    "raynor": ["02-jim-raynor-the-fool-chunk-0",
               "02-jim-raynor-the-fool-chunk-1"],
    "nova":   ["03-nova-terra-the-beautiful-chunk-0",
               "03-nova-terra-the-beautiful-chunk-1"],
}

BACKUP_FILE = CURRENT_DIR / "removed_chunks_backup.json"


def to_serializable(obj):
    """numpy array -> list"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def main():
    chroma_path = str(TASK3_DIR / RELATIVE_CHROMA_PATH)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Чанков до удаления: {collection.count()}")

    all_ids = []
    for ids in ENTITIES_TO_REMOVE.values():
        all_ids.extend(ids)

    existing = collection.get(
        ids=all_ids,
        include=["documents", "metadatas", "embeddings"]
    )

    backup = []
    for i, doc_id in enumerate(existing["ids"]):
        emb = existing["embeddings"][i]
        backup.append({
            "id": doc_id,
            "document": existing["documents"][i],
            "metadata": existing["metadatas"][i],
            "embedding": to_serializable(emb),
        })

    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(backup, f, ensure_ascii=False, indent=2)
    print(f"Бэкап сохранен: {BACKUP_FILE} ({len(backup)} чанков)")

    collection.delete(ids=all_ids)
    print(f"Удалено: {len(all_ids)} чанков")
    print(f"Чанков после удаления: {collection.count()}")


if __name__ == "__main__":
    main()