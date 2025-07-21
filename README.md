# Ask My Course

A full-stack AI-powered question-answering assistant for university students, leveraging Retrieval-Augmented Generation (RAG) to answer questions based on course materials (PDFs, markdown, text files, etc.).

## Features
- Document ingestion and chunking
- Embedding with sentence-transformers
- FAISS vector database for retrieval
- LLM-powered answer generation
- FastAPI backend
- (Optional) Streamlit/Gradio UI
- Production-ready deployment (Docker/Render)

## Project Structure

```
ask_my_course/
├── backend/
│   ├── api/                # FastAPI app and endpoints
│   ├── core/               # Core logic: RAG pipeline, embedding, retrieval
│   ├── models/             # Data models and schemas
│   ├── utils/              # Utility functions (parsing, chunking, etc.)
│   ├── vectorstore/        # FAISS vector DB management
│   ├── main.py             # FastAPI entrypoint
│   └── requirements.txt    # Backend dependencies
├── frontend/               # (Optional) Streamlit/Gradio UI
│   ├── app.py              # UI entrypoint
│   └── requirements.txt    # Frontend dependencies
├── data/                   # Uploaded/processed course materials
├── .env                    # Environment variables
├── Dockerfile              # Docker setup
└── README.md               # Project overview
``` 