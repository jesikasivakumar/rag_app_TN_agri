# 🏛️ Tamil Nadu Government Welfare Schemes AI (RAG Assistant)

An advanced, production-structured Retrieval-Augmented Generation (RAG) assistant built using LangChain, FAISS, and Streamlit. The application processes official knowledge datasets directly from the Tamil Nadu Government portal (`Agriculture & Farmers Welfare Department`) and provides a conversational, memory-retained interface for semantic queries.

---

## 🚀 Key Features
- **Document Loading & Extraction:** Automated web parsing of dynamic scheme registries via `WebBaseLoader`.
- **Intelligent Chunking:** Semantic segmentation using `RecursiveCharacterTextSplitter` to optimize context windows.
- **Local Similarity Search Engine:** Highly efficient, entirely local spatial vector search powered by Meta AI's `FAISS` library.
- **State-of-the-Art Local Embeddings:** High-fidelity semantic text representations using the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face.
- **Contextual Conversation Memory:** Tracks full message history across interactions so you can ask contextual follow-up questions.
- **Advanced UI Dashboard:** Feature-rich interface containing sidebar application telemetry metrics, session management controls, and collapsible source verification expanders.

---

## 🏗️ System Architecture Pipeline

1. **Ingestion (`rag_pipeline.py`):** Loads content from the web ➡️ Splits text to logical segments ➡️ Encodes to vector spaces ➡️ Compiles and saves local FAISS binary index directory.
2. **Retrieval & Interface (`app.py`):** Captures user inputs ➡️ Computes input embedding ➡️ Queries local FAISS matrix index ➡️ Injects matched context chunks + chat memory arrays into custom template prompts ➡️ Evaluates response using `gpt-4o-mini`.

---

## 🛠️ Installation & Setup

### 1. Clone & Project Initialization
Create your clean development workspace directory:
```bash
mkdir tn_rag_app
cd tn_rag_app
