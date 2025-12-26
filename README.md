# LLM-Powered Knowledge Assistant (Mini RAG System)

## Overview

This project implements an **LLM-powered Knowledge Assistant** using a **Retrieval-Augmented Generation (RAG)** architecture.  
The assistant answers user queries by retrieving relevant information from a set of documents and generating **grounded, source-backed responses** using an open-source language model.

The entire system is implemented using **open-source tools** and executed in **Google Colab**, without relying on any proprietary APIs (e.g., OpenAI).

---

## Key Features

- Retrieval-Augmented Generation (RAG) pipeline  
- Semantic search using dense vector embeddings  
- Fully open-source stack (no paid APIs)  
- Source-grounded answers for transparency  
- Interactive user query input using Google Colab widgets  
- Modular and clean pipeline design  

---

## Technology Stack

- **Programming Language:** Python  
- **Execution Environment:** Google Colab  
- **Framework:** LangChain  
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)  
- **Vector Store:** FAISS  
- **Language Model:** FLAN-T5 (`google/flan-t5-base`)  
- **Document Format:** PDF  
- **User Interface:** ipywidgets (Colab-based interactive UI)

