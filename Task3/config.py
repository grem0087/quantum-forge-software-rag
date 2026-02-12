from pathlib import Path

KNOWLEDGE_BASE_DIR = Path("../Task2/knowledge_base")  # папка с .md файлами
CHROMA_PATH = "./chroma_db_ska3ki"           # каталог для ChromaDB

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

CHUNK_SIZE = 950         # символов (~200–250 слов)
CHUNK_OVERLAP = 120      # перекрытие для контекста