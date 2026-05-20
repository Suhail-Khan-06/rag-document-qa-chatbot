# MedRAG — Medical Document Q&A

A retrieval-augmented generation (RAG) application that answers questions from medical PDFs and shows the exact passages used to generate each response.

**Live Demo:** https://huggingface.co/spaces/SuhailKhan06/medrag-document-qa

---

## Overview

MedRAG allows users to upload medical documents such as research papers, clinical guidelines, and drug monographs, then ask questions in natural language.

The system extracts text from uploaded PDFs, converts document chunks into vector embeddings, stores them in ChromaDB, retrieves the most relevant passages, and uses a large language model to generate grounded answers.

To reduce hallucinations, the model is instructed to answer only from the retrieved context. If the requested information is not present in the documents, the application explicitly states that the answer could not be found.

---

## Key Features

- Upload one or more medical PDF documents
- Ask questions in plain English
- Semantic retrieval using vector embeddings
- Source attribution with file names and page numbers
- Persistent ChromaDB vector store
- Local embedding generation with no API cost
- Asynchronous LLM inference using Hugging Face Inference API
- Interactive Streamlit interface
- Deployment on Hugging Face Spaces

---

## System Architecture

```text
Medical PDFs
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings (all-MiniLM-L6-v2)
    ↓
ChromaDB Vector Store
    ↓
Top-k Retrieval
    ↓
Prompt Construction
    ↓
Qwen2.5-72B-Instruct
    ↓
Answer + Supporting Passages
```

---

## Technology Stack

Frontend: Streamlit

RAG Framework: LangChain

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Vector Database: ChromaDB

Large Language Model: Qwen/Qwen2.5-72B-Instruct

PDF Parsing: PyPDFLoader and PyMuPDF

Deployment: Hugging Face Spaces

---

## Why These Technologies?

### all-MiniLM-L6-v2

This model is lightweight, fast, and produces high-quality semantic embeddings while running entirely locally.

### ChromaDB

ChromaDB offers persistent storage and efficient similarity search without requiring external infrastructure.

### Qwen2.5-72B-Instruct

This instruction-tuned model provides strong reasoning and reliable response quality for domain-specific question answering.

### Low Temperature (0.2)

A low temperature setting improves consistency and reduces unnecessary creativity, which is critical in medical applications.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Suhail-Khan-06/medrag-document-qa.git
cd medrag-document-qa
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Linux and macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

Generate a token from:

https://huggingface.co/settings/tokens

---

## Running the Application

```bash
mkdir data
# Place your PDF files inside the data directory
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## Project Structure

```text
medrag-document-qa/
├── src/
│   ├── vector_store.py
│   ├── llm_chain.py
│   ├── document_processor.py
│   └── utils.py
├── data/
├── chroma_db/
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## Example Usage

Example question:

```text
What are the common side effects of metformin?
```

Example answer:

```text
Common side effects of metformin include nausea, diarrhea, abdominal discomfort, and loss of appetite.
```

Source:

```text
metformin_guidelines.pdf — Page 4
```

---

## Configuration

Chunk size: 500 characters

Chunk overlap: 50 characters

Retrieval depth: Top 5 chunks

Temperature: 0.2

Typical response time: 3 to 15 seconds depending on API queue time

---

## Limitations

- Retrieval quality may decrease when answers are spread across multiple distant sections.
- Free Hugging Face inference endpoints can experience variable latency.
- Performance depends on document formatting and chunking strategy.

---

## Future Improvements

- Cross-encoder reranking
- Hybrid retrieval using BM25 and dense embeddings
- Metadata-based filtering
- Evaluation metrics such as Recall@K and MRR
- FastAPI backend and Docker deployment
- User authentication and multi-user support

---

## Use Cases

- Clinical guideline exploration
- Drug information lookup
- Medical research summarization
- Educational study assistance
- Healthcare knowledge retrieval

---

## Resume Description

Developed a medical document question-answering system using Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB, and Qwen2.5-72B, enabling source-grounded answers from uploaded clinical PDFs.

---

## Author

Mohammed Suhail Ahmed Khan

GitHub: https://github.com/Suhail-Khan-06

---

## License

This project is released under the MIT License.
