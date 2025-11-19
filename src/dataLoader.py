from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, CSVLoader


def load_all_documents(data_dir: str) -> List[Any]:
    '''
        Load all supported files from the directory and convert to Langchain document structure
        supported: PDF, EXCEL, CSV, TXT, WORD, JSON
    '''
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path:{data_path}")

    documents = []

    
    # PDF Files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"{len(pdf_files)} PDF files:{[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"Loading pdf file:{pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f" Loaded {len(loaded)} pdf files from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f" failed to load {pdf_file}: {e}") 

  # TXT Files
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"{len(txt_files)} txt files:{[str(f) for f in txt_files]}")

    for txt_file in txt_files:
        print(f"Loading txt file:{txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f" Loaded {len(loaded)} txt files from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f" failed to load {txt_file}: {e}") 

# CSV Files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"{len(csv_files)} csv files:{[str(f) for f in csv_files]}")

    for csv_file in csv_files:
        print(f"Loading csv file:{csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f" Loaded {len(loaded)} csv files from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f" failed to load {csv_file}: {e}") 

    return documents
          
if __name__ == "__main__":
    documents = load_all_documents("./data")
    print(len(documents), 'documents loaded')
    