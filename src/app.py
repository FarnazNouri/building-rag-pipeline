import os
from dataLoader import load_all_documents
from embedding import EmbeddingPipeline
from vectorstore import FaissVectorStore
from search import RAGSearch

if __name__ == "__main__":
    # documents = load_all_documents('./data')
    faiss_vectorStore = FaissVectorStore('faiss_store')
    
    # Load the existing index if it exists
    faiss_path = os.path.join('faiss_store', "faiss_index") 
    meta_path = os.path.join('faiss_store', "metadata.pkl")
    
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        faiss_vectorStore.load()
    else:
        # If no existing index, build from documents
        documents = load_all_documents('./data')
        faiss_vectorStore.build_from_documents(documents)
    
    print(faiss_vectorStore.query('what is attention mechanism?', top_k=3))

    ragsearch = RAGSearch()
    query = "what is attention mechanism?"
    summary = ragsearch.search_and_summarize(query, top_k=3)
    print('summary:', summary)
