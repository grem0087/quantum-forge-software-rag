
from config import KNOWLEDGE_BASE_DIR, CHROMA_PATH, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from chunker import load_and_split_documents
from indexer import get_embedding_function, create_or_get_collection, index_chunks

if __name__ == "__main__":
    all_chunks = load_and_split_documents(KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    
    embedding_function = get_embedding_function(EMBEDDING_MODEL_NAME)
    collection = create_or_get_collection(CHROMA_PATH)
    
    index_chunks(all_chunks, embedding_function, collection)