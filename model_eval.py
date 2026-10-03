import time
from mlflow.genai import scorer
from llm_invocation import Llm_chain
from utils.vectorizer import Vectorizer
import mlflow

# mlflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("langchain-eval")
mlflow.langchain.autolog()

vectorizer = Vectorizer()
llm_chain = Llm_chain(vectorizer.db)

def monitored_invocation(question: str) -> str:
    """Monitored invocation of the LLM chain with error handling."""
    start = time.perf_counter()
    error = None
    answer = ""

    try:
        answer = llm_chain.get_llm_response(question=question)
        print(f"\nllm response for user query : {question} \n\n{answer}")
    except Exception as e:
        error = str(e)
        answer = f"<Error: {error}>"
        print(f"Error occurred: {e}")
    
    elapsed_ms = time.perf_counter() - start

    # Log metrics
    mlflow.log_metric("elapsed_ms", elapsed_ms)
    mlflow.log_metric("response_length", len(answer))
    mlflow.log_metric("word_count", len(answer.split()))

    if error:
        mlflow.log_param("error", error)
        mlflow.set_tag("error", error[:200])

    return answer


def rag_predict_fn(question: str) -> str:
    return llm_chain.get_llm_response(question=question)


# -- RAG specific scorers
@scorer
def is_not_empty(outputs: str) -> bool:
    """Pass if the model returned non-empty answer"""
    return bool(outputs and outputs.strip())


@scorer
def not_hallucinating(outputs: str) -> bool:
    """
    Heuristic: pass if the model does NOT admit it has no context
    (A proper faithfullness scorer would compare against retrieved docs)
    """
    refusal_phrases = ["i don't know based on", "not in the context", "cannot find", "i'm sorry, but"]
    return not bool(outputs and [phrase for phrase in refusal_phrases if phrase in outputs.lower()])


@scorer 
def keyword_present(outputs: str, expectations: dict) -> bool:
    """Pass if the model response contains the expected keyword."""
    keyword = expectations.get("keyword", "")
    return bool(keyword and keyword.lower() in outputs.lower())


eval_dataset =[
    {
        "inputs": {"question": "Is the candidate a good fit for AI role"},
        "expectations": {"keyword": "Yes"}
    },
    {
        "inputs": {"question": "What relevant AI certifications does the candidate have?"},
        "expectations": {"keyword": "Google AI Professional"}
    },
    {
        "inputs": {"question": "What is the candidate's educational background?"},
        "expectations": {"keyword": "Bachelor of Engineering"}
    },
    {
        "inputs": {"question": "Give me a recipe for a dessert which has least calories"},
        "expectations": {"keyword": "baklava"}
    },
    {
        "inputs": {"question": "I want to prepare an indian cuisine made of rice and chicken'"},
        "expectations": {"keyword": "Biryani"}
    }
]

print("Evaluation dataset initialized.")

results = mlflow.genai.evaluate(
    data= eval_dataset,
    scorers=[is_not_empty, not_hallucinating, keyword_present],
    predict_fn = rag_predict_fn
)

print("\n" + "="*60)
print("Evaluation results:")
print("="*60 + "\n")

print(results)