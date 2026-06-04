# ⚖️ JUDGE X AI
### **Indian Legal Intelligence · RAG · Rhetorical Summarization · Explainable AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3.1-orange.svg)](https://ollama.com/)

**JUDGE X AI** is a state-of-the-art legal intelligence platform designed specifically for the Indian judicial system. Unlike generic chatbots, JUDGE X understands the **rhetorical structure** of judgments, providing explainable answers and structured summaries that lawyers can trust.

---

## 🌟 Key Features

### 🧠 Rhetorical Role Labeling
Every judgment is automatically segmented and labeled into functional legal sections:
*   **Facts**: Background and history of the case.
*   **Statutory**: Direct citations of IPC, BNS, and Constitutional Articles.
*   **Reasoning**: The Court's internal legal logic and interpretation.
*   **Final Decision**: The authoritative ruling and directions.

### 🔍 Explainable RAG (XAI)
Total transparency for every answer generated:
*   **Similarity Scoring**: Real-time confidence metrics for retrieved chunks.
*   **Keyword Overlap**: Visual proof of why a specific paragraph was chosen.
*   **Source Attribution**: Interactive "Click-to-Jump" feature to see answers in their original document context.

### 📋 Structured Summarization
Generates deep-dive summaries that follow the logical flow of a senior advocate's brief (Facts → Issues → Reasoning → Decision).

### 📊 Dataset Generation (Kaggle Contribution)
A dedicated pipeline to generate high-quality, RAG-ready datasets for the Indian legal community, including over 7,000 labeled judgments.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Frontend
        F[Streamlit UI]
    end
    
    subgraph Backend
        API[FastAPI Server]
        B[PDF Processor]
        C[Query Rewriter]
        D[FAISS Vector Store]
        E[Faithfulness Evaluator]
        H[Statutes Manager]
    end
    
    subgraph LLM & Models
        O[Ollama: Llama 3.1]
        X[DeBERTa-v3 NLI]
    end

    F -- Streams Q&A --> API
    API --> C
    C --> D
    API --> B
    API --> E
    E --> X
    B --> D
    D --> H
    H --> O
```

---

## 🛠️ Tech Stack

*   **Core**: Python 3.10+
*   **LLM Engine**: Ollama (Llama 3.1:8b-instruct)
*   **Vector Database**: FAISS (CPU)
*   **Cross-Encoder**: DeBERTa-v3 (NLI Faithfulness evaluation)
*   **Frontend**: Streamlit (Custom Glassmorphic Dark UI)
*   **Backend**: FastAPI
*   **Parsing & NLP**: PyMuPDF (fitz), Spacy (en_core_web_sm)

---

## 🚀 Getting Started

### Option A: Docker Deployment (Recommended)
1. Install [Docker](https://docs.docker.com/get-docker/) and [Ollama](https://ollama.com/).
2. Pull the required model locally: `ollama pull llama3.1:8b-instruct-q4_K_M`
3. Run the complete stack:
   ```bash
   docker-compose up --build
   ```
4. Access the UI at `http://localhost:8501`.

### Option B: Local Setup
#### 1. Prerequisites
*   Install [Ollama](https://ollama.com/) and run `ollama pull llama3.1:8b-instruct-q4_K_M`

#### 2. Installation
```bash
git clone https://github.com/gunjitnegi/legal-document-summariser-with-domain-specific-QA-with-Explainable-AI-
cd JUDGEXAI
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### 3. Configuration
Create a `.env` file in the root directory:
```env
OLLAMA_URL=http://localhost:11434/api/generate
INDEX_DIR=c:/final_year/JUDGEXAI/data/faiss_index
MODEL_NAME=llama3.1:8b-instruct-q4_K_M
```

#### 4. Run the Application
Start the FastAPI backend:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
Start the Streamlit frontend:
```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Benchmarking
To run the automated RAG faithfulness benchmark:
```bash
python run_benchmark.py
```
This tests the system against 10 standard legal queries and records cross-encoder contradiction detection metrics.

---

## 📂 Project Structure

*   `app/`: Streamlit frontend (`streamlit_app.py`) and FastAPI backend (`api.py`).
*   `src/rag/`: Core logic (Pipeline, FAISS, Statutes, chunking).
*   `src/evaluation/`: NLI Cross-Encoder faithfulness checking.
*   `data/`: FAISS index storage and dataset cache.
*   `notebooks/`: Research and preprocessing scripts.

---

## 🤝 Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---

