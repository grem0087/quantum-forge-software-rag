from typing import List, Dict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_documents(knowledge_base_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = []
    for md_path in knowledge_base_dir.glob("*.md"):
        with open(md_path, encoding="utf-8") as f:
            text = f.read()

        doc_chunks = splitter.create_documents(
            [text],
            metadatas=[{"source": md_path.name, "title": md_path.stem}]
        )

        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "id": f"{md_path.stem}-chunk-{i}",
                "text": chunk.page_content,
                "metadata": {
                    "source": md_path.name,
                    "title": md_path.stem,
                    "chunk_index": i,
                    "start_index": chunk.metadata.get("start_index", 0),
                }
            })

    print(f"Создано {len(chunks)} чанков")
    return chunks