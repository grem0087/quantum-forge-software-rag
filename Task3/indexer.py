import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pydantic_settings import BaseSettings as PydanticBaseSettings  
from typing import List, Dict

def get_embedding_function(model_name: str):
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},        
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="", 
    )

def create_or_get_collection(chroma_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection(name=collection_name)
        print("Коллекция уже существует, используем её")
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} 
        )
        print("Создана новая коллекция")
    return collection

def index_chunks(chunks: List[Dict], embedding_function, collection):
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Генерация эмбеддингов (batch)
    embeddings = embedding_function.embed_documents(texts)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts  # для возврата текста
    )

    print(f"Загружено {len(chunks)} чанков в ChromaDB")