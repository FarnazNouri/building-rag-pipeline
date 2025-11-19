import os
import faiss
import pickle
import numpy as np
from typing import List, Any
from embedding import EmbeddingPipeline
from sentence_transformers import SentenceTransformer

class FaissVectorStore:
    """
    A vector store implementation using FAISS (Facebook AI Similarity Search) for efficient similarity search.
    This class provides functionality to create, manage, and persist a FAISS vector store with associated
    metadata. It handles document embedding, indexing, and retrieval operations.
    Attributes:
        persis_dir (str): Directory path where the FAISS index and metadata will be persisted.
                         Default is 'Faiss_store'.
        embedding_model (str): Name of the SentenceTransformer model used for generating embeddings.
                              Default is "all-MiniLM-L6-V2".
        chunk_size (int): Size of text chunks for document splitting. Default is 1000.
        chunk_overlap (int): Number of overlapping characters between consecutive chunks. Default is 200.
        model (SentenceTransformer): Loaded sentence transformer model for generating embeddings.
        index (faiss.Index): FAISS index for storing and searching vector embeddings.
                            Initialized as 0 and created when first embeddings are added.
        metadata (List[Dict]): List storing metadata associated with each embedded chunk.
    Methods:
        build_from_documents(documents: List[Any]):
            Processes a list of documents by chunking, embedding, and adding them to the vector store.
            Args:
                documents (List[Any]): List of documents to be processed and added to the store.
            Note:
                - Uses EmbeddingPipeline to chunk and embed documents
                - Automatically saves the index after building
        add_embeddings(embeddings: np.array, metadata: List[Any] = None):
            Adds embeddings and their associated metadata to the FAISS index.
            Args:
                embeddings (np.array): Numpy array of embeddings with shape (n_samples, dimension).
                metadata (List[Any], optional): List of metadata dictionaries corresponding to each embedding.
            Note:
                - Creates the FAISS index on first call based on embedding dimensions
                - Uses L2 (Euclidean) distance metric for similarity search
        save():
            Persists the FAISS index and metadata to disk.
            Note:
                - Saves FAISS index as 'faiss_index' file
                - Saves metadata as 'metadata.pkl' using pickle serialization
                - Both files are stored in the persist_dir directory
    Example:
        >>> vector_store = FaissVectorStore(persist_dir='my_index', embedding_model='all-MiniLM-L6-V2')
        >>> vector_store.build_from_documents(documents)
        >>> vector_store.save()
    """
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

    def build_from_documents(self, documents: List[Any]):
        embeddingPipeline = EmbeddingPipeline()
        chunks = embeddingPipeline.chunk_documents(documents)
        embedded_chunks = embeddingPipeline.embed_chunks(chunks)
        metadata = [{"text":chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embedded_chunks).astype('float32'), metadata)
        self.save()

    def add_embeddings(self, embeddings: np.array, metadata: List[Any]= None):
        if self.index == 0:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        if metadata:
            self.metadata.extend(metadata)

    def save(self):
        faiss_path = os.path.join(self.persis_dir, "faiss_index") 
        meta_path = os.path.join(self.persis_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print("**************************************")
        print('[INFO] Saved Faiss Index and Metadata')

    def load(self):
        faiss_path = os.path.join(self.persis_dir, "faiss_index") 
        meta_path = os.path.join(self.persis_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print("**************************************")
        print('[INFO] Loaded Faiss Index and Metadata')


    def search(self, embedding_query: np.array, top_k: int= 5):
        D, I = self.index.search(embedding_query, top_k)
        result = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            result.append({'index': idx, 'distance': dist, 'metadata': meta})
        return result
    
    def query(self, query_text: str, top_k:int = 5):
        print(f'[INFO] Querying vector store for {query_text}')
        query_embd = self.model.encode([query_text]).astype('float32')
        return self.search(query_embd, top_k=top_k)


