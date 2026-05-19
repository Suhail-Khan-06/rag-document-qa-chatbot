---
title: MedRAG Document QA
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.45.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# MedRAG — Medical Document Q&A

A RAG-powered medical document question-answering chatbot built with:

- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face Inference API
- Streamlit

## Features

- Upload one or more PDF documents
- Automatically chunk and embed text
- Store embeddings in ChromaDB
- Retrieve relevant context using semantic search
- Generate grounded answers with source citations

## Tech Stack

- Python
- LangChain
- ChromaDB
- all-MiniLM-L6-v2
- Hugging Face LLMs
- Streamlit
- Hugging Face Spaces