import os
import sys

# Add the parent directory to sys.path to import loader from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from utils import pdf_loader, json_loader

index_path = "faiss_index"
db: FAISS = None
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small")


class Vectorizer:
    def __init__(self):
        self.index_path = index_path
        self.embeddings = embeddings
        self.db: FAISS = db
        # Load the vector index upon initialization
        self.load_vector_index()


    def get_pdf_doc_chunks(self)->list[Document]:
        docs = pdf_loader.load_pdf_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        return chunks


    def get_json_data_chunks(self)->list[Document]:
        docs:list[Document] = json_loader.get_json_data_as_semantic_text_document()
        return docs
        

    def save_vector_index(self):
        
        pdf_chunks = self.get_pdf_doc_chunks()
        json_chunks = self.get_json_data_chunks()

        self.db = FAISS.from_documents(pdf_chunks + json_chunks, self.embeddings)

        self.db.save_local(self.index_path)

        # embeddings = np.array(db.embeddings).astype(np.float32)
        # print(f"Number of embeddings in the database: {len(db.embeddings)}, and the vector dimension is: {embeddings.shape[1]}")


    def load_vector_index(self):
        if os.path.isdir(self.index_path) and os.path.exists(os.path.join(self.index_path, "index.faiss")):
            try:
                print("\nLoading existing FAISS index...")
                self.db =  FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Recreating the index...")
                self.save_vector_index()
        else:
            print("\nFAISS index not found. Creating a new index...")
            self.save_vector_index()
        
        if self.db is None:
            print("\nLoading FAISS from newly created index...")
            self.db =  FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)

    

if __name__ == "__main__":
    vectorizer = Vectorizer()
    queries = ["What is the candidate's experience with Python?", "I want to prepare an indian cuisine made of rice and chicken"]
    
    # Method 1: Using similarity_search_with_score with score_threshold filter
    for query in queries:
        # Get documents with scores and filter by score_threshold (only scores < 1.0)
        docs_with_scores = vectorizer.db.similarity_search_with_score(
            query, 
            k=3,
            score_threshold=1.0  # Only return documents with score < 1.0
        )
        
        print(f"Top {len(docs_with_scores)} relevant documents for the query '{query}' (score < 1.0):")
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"\nDocument {i+1}")
            print(f"Similarity Score: {score:.4f}")
            print(f"Content preview: {doc.page_content[:150]}")

    