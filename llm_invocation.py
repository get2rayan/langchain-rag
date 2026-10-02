import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from utils.vectorizer import Vectorizer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


model = "openai/gpt-oss-20b"

class langchain_invoke:
    """Class involving langchain paradigm to answer user query"""

    def __init__(self, faiss_index: FAISS):
        self.retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3, "score_threshold": 1.2})

    def format_docs(self, docs: list[Document]) -> str:
        for i, doc in enumerate(docs):
            print(f"\n{i+1}. {doc.page_content}\n")
        return "\n\n".join(doc.page_content for doc in docs)


    def get_llm_response(self, question: str) -> str:
        llm = ChatGroq(model=model, 
                    temperature=0.5, 
                    api_key=os.environ.get("GROQ_API_KEY"))

        template="You are a helpful assistant. Using **only** the below context, give a short concise answer to the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"

        prompt = PromptTemplate.from_template(template=template)

        parser = StrOutputParser()
        
        chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | parser
        )

        result = chain.invoke(question)
        return result


def main():
    vectorizer = Vectorizer()
    lang_chain = langchain_invoke(vectorizer.db)
    
    questions = ["What is the candidate's experience with AI?", "I want a recipe for a dessert which has least calories"]
    for question in questions:
        print(f"\nQuestion: {question}")
        print("llm_response: ", lang_chain.get_llm_response(question), "\n")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
