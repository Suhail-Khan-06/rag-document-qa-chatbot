---
title: MedRAG — Medical Document Q&A
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.45.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# MedRAG — Medical Document Q&A

> Upload medical PDFs and ask grounded, source-attributed questions — powered by RAG, LangChain, and Qwen2.5-72B.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-orange?style=flat-square)](https://huggingface.co/spaces/SuhailKhan06/medrag-document-qa)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.4.1-green?style=flat-square)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.9-purple?style=flat-square)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?style=flat-square)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## Overview

MedRAG is a Retrieval-Augmented Generation (RAG) system designed for medical document question-answering. Users upload one or more PDF documents and ask natural language questions; the system retrieves the most semantically relevant passages and generates a grounded answer using a large language model — citing the exact source chunks used.

Unlike a bare LLM, MedRAG never fabricates information outside the uploaded documents. If the answer is not present, it says so explicitly.

---

## Key Features

- **Source-attributed answers** — every response shows which document chunks were used, with file name and page number
- **Hallucination-resistant** — the LLM is instructed to answer only from retrieved context; out-of-scope questions return a clear "not found" message
- **Free embeddings, no API key needed for ingestion** — uses `all-MiniLM-L6-v2` locally via `sentence-transformers`
- **Persistent vector store** — ChromaDB persists between sessions; re-uploading the same documents does not re-embed
- **Async LLM inference** — non-blocking calls to `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API
- **Medical domain focus** — prompt engineering tuned for clinical and research document terminology

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│                                                                 │
│  PDF Files  ──►  PyPDFLoader  ──►  RecursiveCharacterText       │
│                                    Splitter (500/50)            │
│                                          │                      │
│                                          ▼                      │
│                              all-MiniLM-L6-v2 Embeddings        │
│                              (sentence-transformers, local)     │
│                                          │                      │
│                                          ▼                      │
│                              ChromaDB (persistent store)        │
└───────────────────────────────────────────────────────┬─────────┘
                                                        │
                                                        │ at query time
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL + GENERATION                   │
│                                                                 │
│  User Query  ──►  Embed Query  ──►  Cosine Similarity Search    │
│                                          │                      │
│                                     Top-5 Chunks                │
│                                          │                      │
│                                          ▼                      │
│                              RAG Prompt Construction            │
│                         (context + query + system role)         │
│                                          │                      │
│                                          ▼                      │
│                         Qwen2.5-72B-Instruct (async)            │
│                         HuggingFace Inference API               │
│                                          │                      │
│                                          ▼                      │
│                          Answer + Source Attribution            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer           | Technology                                                |
| --------------- | --------------------------------------------------------- |
| Frontend        | Streamlit 1.45                                            |
| Orchestration   | LangChain Community 0.4 + LangChain Core                  |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2` (local, free)    |
| Vector Store    | ChromaDB 1.5 (persistent)                                 |
| LLM             | `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API |
| PDF Parsing     | PyPDF + PyMuPDF                                           |
| Async Inference | `huggingface_hub.AsyncInferenceClient`                    |
| Deployment      | HuggingFace Spaces                                        |

---

## Results

| Metric              | Value                        |
| ------------------- | ---------------------------- |
| Embedding model     | `all-MiniLM-L6-v2` (384-dim) |
| Chunk size          | 500 tokens, 50-token overlap |
| Retrieval strategy  | Cosine similarity, top-k=5   |
| LLM                 | Qwen2.5-72B-Instruct         |
| Max response tokens | 512                          |
| Temperature         | 0.2 (factual, low variance)  |
| Avg latency (local) | ~3–6 seconds per query       |

---

## Demo

> **Live app:** [huggingface.co/spaces/SuhailKhan06/medrag-document-qa](https://huggingface.co/spaces/SuhailKhan06/medrag-document-qa)

_Add a demo GIF here: record a 10–15 second screen capture of uploading a PDF and asking a question. Recommended tool: [Loom](https://loom.com) or [ShareX](https://getsharex.com)._

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Suhail-Khan-06/medrag-document-qa
cd medrag-document-qa

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your HuggingFace API token
echo "HUGGINGFACEHUB_API_TOKEN=hf_your_token_here" > .env

# 5. Add PDF files to the data/ directory
mkdir data
# copy your PDFs into data/

# 6. Run the Streamlit app
streamlit run app.py
```

Get a free HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Project Structure

```
medrag-document-qa/
├── src/
│   ├── vector_store.py       # PDF loading, chunking, embedding, ChromaDB
│   ├── llm_chain.py          # RAG prompt, async LLM inference, source retrieval
│   ├── document_processor.py # Document preprocessing utilities
│   └── utils.py              # Shared helper functions
├── data/                     # Place PDF files here (gitignored)
├── chroma_db/                # Persisted ChromaDB store (gitignored)
├── app.py                    # Streamlit application entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## How It Works

**1. Ingestion** — PDFs are loaded with `PyPDFLoader`, split into 500-token overlapping chunks using `RecursiveCharacterTextSplitter`, and embedded with `all-MiniLM-L6-v2` (a lightweight but accurate sentence transformer). Embeddings are stored in a persistent ChromaDB collection.

**2. Retrieval** — At query time, the user's question is embedded with the same model and compared against stored chunk embeddings using cosine similarity. The top-5 most relevant chunks are retrieved.

**3. Generation** — Retrieved chunks are assembled into a structured RAG prompt and sent to `Qwen2.5-72B-Instruct` via the HuggingFace Inference API (async). The model is instructed to answer strictly from the provided context.

**4. Attribution** — The response is returned alongside the source chunks, including file name and page number, so users can verify every claim against the original document.

---

## Design Decisions

**Why `all-MiniLM-L6-v2` for embeddings?**
It runs locally with no API key, loads in under 2 seconds, and achieves strong semantic retrieval quality for its size (22M parameters). This keeps the ingestion pipeline fully free and offline.

**Why `Qwen2.5-72B-Instruct`?**
It is one of the strongest open-weight instruction-tuned models available on the free HuggingFace Inference API tier, with demonstrated strong performance on knowledge-intensive and medical reasoning tasks.

**Why temperature 0.2?**
Medical Q&A demands factual precision over creative variation. Low temperature reduces hallucination risk in the generation step on top of the retrieval constraint.

---

## Limitations

- Requires a HuggingFace API token for LLM inference (free tier available)
- Response latency depends on HuggingFace Inference API queue times
- Retrieval quality is limited by chunk size; very long answers spanning multiple pages may be partially missed
- Currently English-language documents only

---

## Future Work

- [ ] Add MMR (Maximal Marginal Relevance) retrieval to reduce redundant source chunks
- [ ] Add a re-ranking step (cross-encoder) for improved retrieval precision
- [ ] Support DOCX and plain text input formats
- [ ] Add conversation memory for multi-turn Q&A
- [ ] Evaluate retrieval with RAGAS framework

---

## Related Work

This project builds on the RAG paradigm introduced in [Lewis et al. (2020)](https://arxiv.org/abs/2005.11401) and uses the retrieval architecture described in the [LangChain documentation](https://python.langchain.com/docs/use_cases/question_answering/).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Mohammed Suhail Ahmed Khan**
B.Tech Computer Science and Engineering, IIIT Kottayam (Expected May 2027)

[![GitHub](https://img.shields.io/badge/GitHub-Suhail--Khan--06-black?style=flat-square&logo=github)](https://github.com/Suhail-Khan-06)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/mohammed-suhail-ahmed-khan)
