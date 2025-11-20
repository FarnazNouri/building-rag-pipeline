import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from vectorstore import FaissVectorStore

load_dotenv()

# Load API key from environment
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")

class RAGSearch:
    def __init__(self, persist_dir: str = 'Faiss_store', embedding_model:str = "all-MiniLM-L6-V2"):
        self.vectorStore = FaissVectorStore(persist_dir, embedding_model)
        #Load or Build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss_index") 
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from dataLoader import load_all_documents
            docs = load_all_documents('./data')
            self.vectorStore.build_from_documents(docs)
        else:
            self.vectorStore.load()


        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",  # Proper LLM model name
            temperature=0.1
        )
        print('[INFO] *****Groq model initiated****')

    def search_and_summarize(self, query:str , top_k:int=5) -> str:
        results = self.vectorStore.query(query, top_k=top_k)
        texts = [r['metadata'].get("text","") for r in results if r['metadata']]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant document found"
        prompt = f"""Summarize the follwoing context for the query: '{query}'\n\nContext:\n\n'{context}"""
        response = self.llm.invoke(prompt)
        return response.content

        