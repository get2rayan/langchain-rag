# langchain-rag

A LangChain Retrieval-Augmented Generation (RAG) demo that retrieves from PDF documents and recipe data using a local FAISS vector store, then generates answers with Groq.

## Project Structure

```
.
├── data/
│   └── recipes_dataset.json  # Recipe records; add source PDFs here
├── faiss_index/              # Generated locally; ignored by Git
├── utils/
│   ├── __init__.py          # Utility package
│   ├── json_loader.py        # Converts recipe records to searchable text
│   ├── pdf_loader.py         # Loads PDFs from data/
│   └── vectorizer.py         # Builds or loads the FAISS index
├── llm_invocation.py         # Runs the interactive-style example queries
├── model_eval.py             # Runs the MLflow evaluation
├── .env.template             # API key and evaluation setting examples
├── .python-version            # Python version for uv
├── pyproject.toml            # Dependencies and project metadata
└── uv.lock                   # Locked dependency versions
```

## Requirements and Setup

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for embeddings and a Groq API key for generation

Install dependencies and create the local environment file:

```bash
uv sync
cp .env.template .env
```

Edit `.env` and replace the placeholder values for `OPENAI_API_KEY` and `GROQ_API_KEY`. The PDF loader loads this file when the application starts. Do not commit `.env`.

## Data and Index

Place PDF files in `data/`; the PDF loader processes every `*.pdf` file in that directory. The candidate-focused demo questions require relevant candidate documents, while the checked-in `recipes_dataset.json` supplies recipe content. Recipe text includes each recipe's name, cuisine, cook time, ingredients, and nutrition; instructions are not included in the indexed text.

PDF pages are split into chunks of 500 characters with 100 characters of overlap. Embeddings use OpenAI's `text-embedding-3-small` model. The FAISS index is saved under `faiss_index/`, which is ignored by Git. If no index exists, the vectorizer builds one from the available PDFs and JSON data. It reuses an existing index on later runs; after changing source data, remove `faiss_index/` to rebuild it.

## Run the RAG Demo

```bash
uv run llm_invocation.py
```

The script runs two example questions, retrieves up to three relevant documents for each, and prints the generated answers. Generation uses Groq's `openai/gpt-oss-20b` model.

## Model Evaluation

Start the MLflow tracking server in one terminal:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Then run the evaluation in another terminal:

```bash
uv run model_eval.py
```

The script evaluates five fixed questions and logs results to the `langchain-eval` experiment at `http://localhost:5000`. Its scorers check for a non-empty response, selected refusal phrases, and an expected keyword. The refusal-phrase check is a simple heuristic, not a groundedness or factuality measurement. Open the MLflow UI at `http://localhost:5000` to review runs.

MLflow trace validation can make a preliminary prediction on the first sample, which adds an LLM call. The supplied `.env.template` sets `MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION=True` to skip it. If that variable is not set in your environment, set it when running the evaluation:

```bash
MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION=True uv run model_eval.py
```

## License

MIT