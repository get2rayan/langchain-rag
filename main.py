import os
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from utils.vectorizer import load_vector_index
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


model = "groq/compound-mini"

def get_llm_response(question: str, relevant_docs: list[Document]) -> str:
    llm = ChatGroq(model=model, 
                   temperature=0.5, 
                   api_key=os.environ.get("GROQ_API_KEY"))

    template="You are a helpful assistant. Use the following context to give a short concise answer to the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"

    prompt = PromptTemplate.from_template(template=template)

    parser = StrOutputParser()
    
    chain = prompt | llm | parser

    result = chain.invoke({"context": "\n".join([doc.page_content for doc in relevant_docs]), "question": question})

    return result



def main():
    question = "What is the candidate's experience with Python?"
    db = load_vector_index()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question) 
    print("get_llm_response: ", get_llm_response(question, relevant_docs))



if __name__ == "__main__":
    main()
