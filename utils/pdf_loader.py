from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

data_dir = Path(__file__).parent.parent / "data"


def load_pdf_documents(file_path: Path = None) -> list:
    """Load documents from all PDF files in the data directory, or from a specific file if provided."""
    docs = []
    
    if file_path:
        # Load a specific file if provided
        if file_path.exists():
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        else:
            print(f"File {file_path} does not exist.")
    else:
        # Load all PDF files from the data directory
        if data_dir.exists():
            pdf_files = list(data_dir.glob("*.pdf"))
            if not pdf_files:
                print(f"No PDF files found in {data_dir}")
                return []
            
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(pdf_file)
                    loaded_docs = loader.load()
                    docs.extend(loaded_docs)
                    print(f"Loaded {len(docs)} documents from {pdf_file.name}")
                except Exception as e:
                    print(f"Error loading {pdf_file.name}: {e}")
        else:
            print(f"Data directory {data_dir} does not exist.")
    
    return docs


if __name__ == "__main__":
    docs = load_pdf_documents()
    print(f"Loaded {len(docs)} documents from data directory.")
    for i, doc in enumerate(docs):
        print(f"Document {i+1} content preview: {doc.page_content[:200]}")  # Print first 200 characters of each document
    
