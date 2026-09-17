import os
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from utils.vectorizer import load_vector_index
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


model = "groq/compound-mini"

def format_docs(docs: list[Document]) -> str:
    return "\n".join(doc.page_content for doc in docs)


def get_llm_response(question: str, retriever) -> str:
    llm = ChatGroq(model=model, 
                   temperature=0.5, 
                   api_key=os.environ.get("GROQ_API_KEY"))

    template="You are a helpful assistant. Use the following context to give a short concise answer to the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"

    prompt = PromptTemplate.from_template(template=template)

    parser = StrOutputParser()
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )

    result = chain.invoke(question)

    return result



def main():
    question = "What is the candidate's experience with Python?"
    db = load_vector_index()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("get_llm_response: ", get_llm_response(question, retriever))



if __name__ == "__main__":
    main()
