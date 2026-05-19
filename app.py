"""
Streamlit frontend for MedRAG — Medical Document Q&A Chatbot.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st
from langchain_core.documents import Document

from src.llm_chain import ask_question
from src.vector_store import build_vector_store

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="MedRAG — Medical Document Q&A",
    page_icon="🩺",
    layout="wide",
)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------
st.title("🩺 MedRAG — Medical Document Q&A")
st.caption("Upload medical PDFs and ask grounded questions based on the document content.")

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_button = st.button("Process Documents", type="primary")

    if st.session_state.chunk_count > 0:
        st.success(f"Processed {st.session_state.chunk_count} chunks.")

    # Process uploaded PDFs
    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing documents..."):
                # Clear existing PDFs
                for existing_pdf in DATA_DIR.glob("*.pdf"):
                    existing_pdf.unlink()

                # Save uploaded files to data/
                for uploaded_file in uploaded_files:
                    file_path = DATA_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Build vector store
                st.session_state.vector_store = build_vector_store(
                    pdf_directory=str(DATA_DIR)
                )

                # Estimate chunk count by querying all stored docs
                try:
                    collection = st.session_state.vector_store.get()
                    st.session_state.chunk_count = len(
                        collection.get("documents", [])
                    )
                except Exception:
                    st.session_state.chunk_count = 0

                # Reset chat history when new documents are processed
                st.session_state.chat_history = []

            st.success("Documents processed successfully!")

# -------------------------------------------------------------------
# Display chat history
# -------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):
            with st.expander("View sources"):
                for i, doc in enumerate(message["sources"], start=1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")

                    st.markdown(f"**Source {i}**")
                    st.markdown(f"- File: `{source}`")
                    st.markdown(f"- Page: `{page}`")
                    st.write(doc.page_content)
                    st.divider()

# -------------------------------------------------------------------
# Chat input
# -------------------------------------------------------------------
question = st.chat_input("Ask a question about the uploaded documents...")

if question:
    # Check if documents are processed
    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents first.")
        st.stop()

    # Add user message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                result = ask_question(
                    query=question,
                    vector_store=st.session_state.vector_store,
                    k=5,
                )

                answer = result["answer"]
                sources: List[Document] = result["sources"]

                # Display answer
                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("View sources"):
                        for i, doc in enumerate(sources, start=1):
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "N/A")

                            st.markdown(f"**Source {i}**")
                            st.markdown(f"- File: `{source}`")
                            st.markdown(f"- Page: `{page}`")
                            st.write(doc.page_content)
                            st.divider()

                # Save assistant response
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:
                error_message = f"Error: {e}"
                st.error(error_message)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )