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

Upload a medical PDF, ask a question, get an answer with the exact passage it came from.

**[Try it live →](https://huggingface.co/spaces/SuhailKhan06/medrag-document-qa)**

---

## What it does

You drop in one or more medical PDFs — research papers, clinical guidelines, drug monographs — and ask questions in plain English. The app finds the most relevant passages, hands them to an LLM with strict instructions to answer only from what's there, and shows you both the answer and the source chunks it used.

If the answer isn't in the document, it says so instead of making something up. That constraint matters a lot in a medical context.

---

## How it works

```
PDF files
   │
   ▼
PyPDFLoader → RecursiveCharacterTextSplitter (500 tokens, 50 overlap)
   │
   ▼
all-MiniLM-L6-v2 embeddings  (local, no API key needed)
   │
   ▼
ChromaDB  ←──────────────────────────────────────────┐
   │                                                  │
   │  at query time                                   │
   ▼                                                  │
embed query → cosine similarity → top-5 chunks ───────┘
   │
   ▼
RAG prompt → Qwen2.5-72B-Instruct  (HuggingFace Inference API, async)
   │
   ▼
Answer + source attribution  (file, page, passage)
```

The embedding step runs locally so ingestion is completely free. Only the generation call hits an external API.

I used `all-MiniLM-L6-v2` because it's fast, runs without a GPU, and gives genuinely good semantic retrieval quality for its size. Temperature is set to 0.2 — medical Q&A needs precision over creativity.

---

## Stack

|               |                                          |
| ------------- | ---------------------------------------- |
| Frontend      | Streamlit                                |
| Orchestration | LangChain                                |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store  | ChromaDB (persistent)                    |
| LLM           | `Qwen/Qwen2.5-72B-Instruct`              |
| PDF parsing   | PyPDF + PyMuPDF                          |
| Deployment    | HuggingFace Spaces                       |

---

## Running locally

```bash
git clone https://github.com/Suhail-Khan-06/medrag-document-qa
cd medrag-document-qa
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
mkdir data
# drop your PDFs into data/
streamlit run app.py
```

---

## Project structure

```
medrag-document-qa/
├── src/
│   ├── vector_store.py       # ingestion, chunking, ChromaDB
│   ├── llm_chain.py          # retrieval, prompt construction, async LLM call
│   ├── document_processor.py
│   └── utils.py
├── data/                     # your PDFs go here  (gitignored)
├── chroma_db/                # persisted vector store  (gitignored)
├── app.py                    # Streamlit entry point
└── requirements.txt
```

---

## Limitations worth knowing

Retrieval quality drops on documents where the answer is spread across multiple non-adjacent sections — the chunking strategy doesn't handle that well. A re-ranker or MMR retrieval would help. The HuggingFace free inference tier also has queue times that vary a lot, so response latency isn't consistent.

---

Built by [Mohammed Suhail Ahmed Khan](https://github.com/Suhail-Khan-06)
