import chromadb.utils.embedding_functions as embedding_functions
import chromadb
from typing import Optional
import os
import dotenv


dotenv.load_dotenv()

class VectorDB:
    def __init__(self, database_path: str, collection_name: str):
        self._collection_name = collection_name
        self._database_path = database_path

        self._embed_func = embedding_functions.OpenAIEmbeddingFunction(
            api_type="azure",
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("API_VERSION"),
            model_name=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
        )

        self._client = chromadb.PersistentClient(path=self._database_path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_func
        )

    def batch_add(self, texts: list[str], metadatas: Optional[list[dict]] = None):
        current_num = len(self._collection.get()["documents"]) if self._collection.get()["documents"] else 0
        self._collection.add(
            ids=[f"{i:03d}" for i in range(current_num, current_num + len(texts))],
            documents=texts,
            metadatas=metadatas
        )
        
    def _query_by_text(
        self, 
        query_text: str, 
        n_results: int = 10, 
        where: Optional[dict] = None
    ) -> dict:
        return self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["metadatas", "embeddings", "documents", "distances"],
        )
    
    def query_l2(self, query_text: str, n_results: int) -> dict:
        return self._query_by_text(
            query_text,
            n_results=n_results,
            where={"level": "Category2"}
        )
    
    def query_l3(self, query_text: str, n_results: int) -> dict:
        return self._query_by_text(
            query_text,
            n_results=n_results,
            where={"level": "Category3"}
        )