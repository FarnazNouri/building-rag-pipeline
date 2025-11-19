from dataLoader import load_all_documents
from embedding import EmbeddingPipeline
from vectorstore import FaissVectorStore

if __name__ == "__main__":
    documents = load_all_documents('./data')
    faiss_vectorStore = FaissVectorStore('faiss_store')
    faiss_vectorStore.build_from_documents(documents)
    print(faiss_vectorStore.query('what is attention mechanism?', top_k=3))