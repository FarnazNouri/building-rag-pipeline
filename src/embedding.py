from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from dataLoader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name:str = 'all-MiniLM-L6-V2', 
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"{model_name} embedding model loaded")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len, 
            separators=["\n\n", "\n", ",", ".", "  ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] {len(documents)} documents splitted into {len(chunks)} chunks")
        return chunks
    
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        text = [chunk.page_content for chunk in chunks]
        print("===================")
        print(f"[INFO] Generating embedding for {len(text)} chunks")
        embedding = self.model.encode(text, show_progress_bar=True)
        print(f"Embedding shape: {embedding.shape} ")
        return embedding
    

if __name__ == "__main__":
    documents = load_all_documents('./data')
    embeddingPipeline = EmbeddingPipeline()
    chunks = embeddingPipeline.chunk_documents(documents)
    embed = embeddingPipeline.embed_chunks(chunks)
