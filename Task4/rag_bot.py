import sys
from pathlib import Path

TASK3_DIR = Path(__file__).resolve().parent.parent / "Task3"
sys.path.append(str(TASK3_DIR))

from config import CHROMA_PATH as RELATIVE_CHROMA_PATH, EMBEDDING_MODEL_NAME, COLLECTION_NAME

import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class KopruluRAGBot:  

    def __init__(
        self,
        chroma_path: str = None,
        llm_model: str = "mistral:latest",
        top_k: int = 5,
        temperature: float = 0.1,
    ):
        self.chroma_path = chroma_path or str(TASK3_DIR / RELATIVE_CHROMA_PATH)
        print(f"[INIT] ChromaDB: {self.chroma_path}")

        # Эмбеддер GPU
        self.embedding_function = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="",
        )

        # Создаём LangChain-обёртку над Chroma (это ключевой шаг!)
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_function
        )

        print(f"[INIT] Документов в коллекции: {self.vectorstore._collection.count()}")

        # Retriever — теперь из vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        # LLM
        self.llm = OllamaLLM(model=llm_model, temperature=temperature)

        # Промпт с Few-shot и CoT
        self.prompt_template = PromptTemplate.from_template(
            """System: Ты эксперт по копрулуским легендам и сказкам сектора. 
Отвечай ТОЛЬКО на основе предоставленного контекста. 
Сначала размышляй шаг за шагом (Chain-of-Thought), затем делай чёткий вывод. 
Если в контексте нет ответа — честно кратко скажи "Я не знаю" или "В моей базе знаний нет такой информации", не выполняя рассуждений.

Примеры (Few-shot):
Q: Кто такой мармырь?
A: Я этого не знаю

Q: Кто такая Загра Бродмать?
A: 1. В контексте Загра описана как древняя бродмать роя.
2. Она обитает в мобильном улье на органических ножках.
3. Может помочь или поглотить гостя.
Вывод: Загра — древняя бродмать, живущая в подвижном улье на ножках-крыльях.

Q: Как победить Абатура Бессмертного?
A: 1. Абатур — эволюционный мастер, чья сущность спрятана в яйце.
2. Нужно найти и разрушить это яйцо в глубинах Чара.
3. Герой проходит три мира для этого.
Вывод: Разрушить главное яйцо эволюции в недрах планеты Чар.

Контекст из базы знаний:
{context}

Вопрос пользователя: {question}

Ответ (сначала шаги размышлений, затем чёткий вывод):"""
        )

        # RAG-цепочка
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

        print("[INIT] RAG-бот готов (Ministral 8B, GPU)")

    def ask(self, query: str) -> str:
        if not query.strip():
            return "Вопрос пустой."

        try:
            result = self.qa_chain.invoke({"query": query})
            answer = result["result"]
            sources = result["source_documents"]

            output = f"Ответ:\n{answer}\n\n"
            
            return output
        except Exception as e:
            return f"[ERROR] {str(e)}"