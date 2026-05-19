"""
LLM response generation for the RAG pipeline.

This module:
1. Retrieves the top-k most relevant chunks from ChromaDB
2. Builds a RAG prompt
3. Sends the prompt to Hugging Face Inference API
4. Returns the generated answer and source chunks used
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def build_prompt(context: str, query: str) -> str:
    """
    Construct the RAG prompt.
    """
    return f"""
You are a helpful medical document assistant.

Answer ONLY using the information provided in the context below.
If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{query}

Answer:
""".strip()


async def generate_answer(
    query: str,
    vector_store: Chroma,
    k: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Generate an answer using retrieved document chunks and Hugging Face Inference API.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    source_documents: List[Document] = vector_store.similarity_search(
        query=query,
        k=k,
    )

    if not source_documents:
        return {
            "answer": "I could not find any relevant information in the document.",
            "sources": [],
        }

    context = "\n\n".join(doc.page_content for doc in source_documents)
    prompt = build_prompt(context=context, query=query)

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN or HF_TOKEN not found in .env file."
        )

    try:
        await asyncio.sleep(1)

        client = AsyncInferenceClient(api_key=hf_token)

        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful medical document assistant. "
                        "Answer only from the provided context. "
                        "If the answer is not in the context, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        answer = response.choices[0].message.content.strip()

        if not answer:
            answer = "The model returned an empty response."

        return {
            "answer": answer,
            "sources": source_documents,
        }

    except Exception as exc:
        raise RuntimeError(f"Failed to generate answer: {exc}") from exc


def ask_question(
    query: str,
    vector_store: Chroma,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Synchronous wrapper around the async function.
    """
    return asyncio.run(
        generate_answer(
            query=query,
            vector_store=vector_store,
            k=k,
        )
    )


if __name__ == "__main__":
    from src.vector_store import build_vector_store

    try:
        print("Loading vector store...")
        db = build_vector_store(pdf_directory="data")

        question = "What are the symptoms of Parkinson's disease?"
        print(f"\nQuestion: {question}")

        result = ask_question(question, db)

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nSources Used:\n")
        for i, doc in enumerate(result["sources"], start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            print(f"--- Source {i} ---")
            print(f"File: {source}")
            print(f"Page: {page}")
            print(doc.page_content[:300])
            print()

    except Exception as e:
        print(f"Error: {e}")