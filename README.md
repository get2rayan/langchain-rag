# langchain-rag

A Retrieval-Augmented Generation (RAG) application built with LangChain.

## Project Structure

```
langchain-rag/
├── data/                    # PDF and other data files
├── faiss_index/             # FAISS vector store index (generated)
├── main.py                  # Main entry point for RAG queries
├── loader.py                # PDF document loader
├── pyproject.toml           # Project configuration
├── utils/                   # Utility modules
│   ├── __init__.py          # Package init
│   ├── vectorizer.py        # Vector store operations
│   └── loader.py            # Document loading utilities
├── .env                     # Environment variables (API keys)
├── .gitignore               # Git ignore rules
├── uv.lock                  # uv lock file
└── README.md                # This file
```

## Features

- **PDF Document Loading**: Loads and processes PDF resumes using `PyPDFLoader`
- **Vector Search**: Creates and searches FAISS vector store for semantic similarity
- **RAG Pipeline**: Combines document retrieval with LLM generation
- **Modular Design**: Separate concerns with dedicated modules

## Prerequisites

- Python 3.13+
- API keys for:
  - OpenAI (for embeddings)
  - Groq (for LLM inference)

## Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up environment variables:
   ```bash
   cp .env.template .env
   # Add your API keys to .env
   ```

3. Run the application:
   ```bash
   uv run main.py
   ```

## Usage

### Loading Documents

The `loader.py` module can load PDF documents from the `data/` directory. By default, it loads **all PDF files** found in the data folder:

```bash
uv run utils/loader.py
```

Or from the project root:

```bash
uv run python -m utils.loader
```

**Output example:**
```
Loaded NP_Resume.pdf
Loaded 2 documents from data directory.
Document 1 content preview: PAGE 1 OF 2 ...
Document 2 content preview: PAGE 2 OF 2 ...
```

**To load a specific file**, pass the path as an argument:

```bash
uv run python -c "from utils.loader import load_documents; docs = load_documents(Path('data/specific_file.pdf'))"
```

The data directory can contain multiple PDF files, and the loader will process all of them, making it easy to add new documents without modifying the code.

### Running the Main Application

```bash
uv run main.py
```

This will:
1. Load and vectorize documents
2. Perform similarity search
3. Get LLM response to queries

## Configuration

See `pyproject.toml` for project dependencies and configuration.

## License

MIT