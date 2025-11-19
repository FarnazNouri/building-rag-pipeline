import os
import faiss
import pickle
import numpy as np
from typing import List, Any
from embedding import EmbeddingPipeline
from sentence_transformers import SentenceTransformer

class FaissVectorStore:
    def __init__(self, persist_dir: str = 'Faiss_store', embedding_model:str = "all-MiniLM-L6-V2",
                  chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persis_dir = persist_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)
        os.makedirs(persist_dir, exist_ok=True)
        self.index = 0
        self.metadata = []
        print(f"[INFO] Embedding model loaded: {embedding_model}")

        


