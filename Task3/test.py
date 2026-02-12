import chromadb
from config import CHROMA_PATH, EMBEDDING_MODEL_NAME
from indexer import get_embedding_function

def test_query(query: str, n_results: int = 3):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="koprulu_myths")
    
    embedding_function = get_embedding_function(EMBEDDING_MODEL_NAME)
    query_embedding = embedding_function.embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nРезультаты для запроса '{query}':")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"[{dist:.4f}] {meta['title']} (chunk {meta['chunk_index']}): {doc[:180]}...")

if __name__ == "__main__":
    queries = [
        "Как победить Абатура Бессмертного?",
        "Что делает Феникс Серый Волк для героя?"
    ]
    
    for q in queries:
        test_query(q)