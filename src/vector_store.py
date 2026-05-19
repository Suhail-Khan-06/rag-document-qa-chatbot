"""
Vector store creation and retrieval utilities for the RAG pipeline.

This module:
1. Loads all PDF files from a directory
2. Splits them into overlapping chunks
3. Generates embeddings using all-MiniLM-L6-v2
4. Stores embeddings in a persistent ChromaDB vector store
5. Returns the initialized Chroma vector store
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def build_vector_store(
    pdf_directory: str,
    persist_directory: str = "chroma_db",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Chroma:
    """
    Build a persistent Chroma vector store from all PDF files in a directory.
    """
    pdf_path = Path(pdf_directory)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_directory}")

    if not pdf_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {pdf_directory}")

    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {pdf_directory}")

    documents: List[Document] = []

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            if docs:
                documents.extend(docs)

        except Exception as exc:
            print(f"Warning: Failed to load {pdf_file.name}: {exc}")

    if not documents:
        raise ValueError("No readable text could be extracted from the PDF files.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting produced zero chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        return vector_store

    except Exception as exc:
        raise RuntimeError(f"Failed to create vector store: {exc}") from exc


def retrieve_relevant_chunks(
    vector_store: Chroma,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for a query.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    return vector_store.similarity_search(query=query, k=k)


if __name__ == "__main__":
    try:
        print("Building vector store...")
        db = build_vector_store(pdf_directory="data")

        print("Vector store created successfully!")

        print("\nTesting retrieval...\n")
        results = retrieve_relevant_chunks(
            vector_store=db,
            query="What are the symptoms of Parkinson's disease?",
            k=3,
        )

        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            print(f"\n--- Result {i} ---")
            print(f"Source: {source}")
            print(f"Page: {page}")
            print(doc.page_content[:500])

    except Exception as e:
        print(f"Error: {e}")