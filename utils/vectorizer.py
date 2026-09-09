import os
import sys

# Add the parent directory to sys.path to import loader from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from utils import loader

index_path = "faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db:FAISS = None


def save_vector_index():
    docs = loader.load_documents()

    if not docs:
        print("No documents loaded. Please check the 'data' directory for the test file.")
        exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embed = embeddings
    db = FAISS.from_documents(chunks, embed)

    db.save_local("faiss_index")

    # embeddings = np.array(db.embeddings).astype(np.float32)
    # print(f"Number of embeddings in the database: {len(db.embeddings)}, and the vector dimension is: {embeddings.shape[1]}")


def load_vector_index():
    global db
    if os.path.isdir(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
        try:
            print("Loading existing FAISS index...")
            db =  FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Recreating the index...")
            save_vector_index()
    else:
        print("FAISS index not found. Creating a new index...")
        save_vector_index()
    
    if db is None:
        print("Loading FAISS from newly created index...")
        db =  FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    return db
    

if __name__ == "__main__":
    load_vector_index()
    query = "What is the candidate's experience with Python?"
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query) 

    print(f"Top {len(relevant_docs)} relevant documents for the query '{query}':")
    for i, doc in enumerate(relevant_docs):
        print(f"\nDocument {i+1} content preview: {doc.page_content[:200]}")  # Print first 200 characters of each document 
