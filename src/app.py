from dataLoader import load_all_documents
from embedding import EmbeddingPipeline

if __name__ == "__main__":
    documents = load_all_documents('./data')
    embeddingPipeline = EmbeddingPipeline()
    chunkVectors = embeddingPipeline.chunk_documents(documents)
    embedded_chunks = embeddingPipeline.embed_chunks(chunkVectors)