# ⚖️ Legal Contract Intelligence Platform

An AI-powered Legal Contract Intelligence Platform that automates contract analysis using Machine Learning, Natural Language Processing (NLP), Semantic Search, and Large Language Models (LLMs).

The platform enables users to classify legal clauses, analyze contract risks, search similar contracts, and interact with contracts through an AI-powered legal assistant.

---

## 🚀 Features

### 📑 Clause Classification

Automatically classifies legal clauses into 36 different categories including:

* Governing Law
* Anti-Assignment
* Non-Compete
* License Grant
* Change of Control
* Exclusivity
* Termination Clauses
* Liability Clauses
* Warranty Clauses

and many more.

---

### ⚠️ Risk Analysis Engine

Analyzes contracts and identifies potentially risky clauses.

Risk factors include:

* Non-Compete Clauses
* Change of Control Clauses
* Uncapped Liability
* Exclusivity Restrictions
* Assignment Restrictions

Provides:

* Risk Score
* Risk Level (Low / Medium / High)
* Detected Risk Factors

---

### 🔍 Similar Contract Search

Uses Sentence Transformers and Semantic Similarity Search to retrieve contracts that are most similar to a user-provided contract.

Capabilities:

* Contract-to-Contract Similarity
* Semantic Search
* Top-K Similar Contract Retrieval

---

### 🤖 Legal AI Assistant

Built using:

* RAG (Retrieval-Augmented Generation)
* FAISS Vector Database
* Groq LLMs

Allows users to:

* Ask legal questions
* Retrieve relevant contract clauses
* Generate context-aware legal responses

---

## 📊 Dataset Information

| Metric            | Value  |
| ----------------- | ------ |
| Contracts         | 509    |
| Legal Clauses     | 5,694  |
| Clause Categories | 36     |
| Contract Chunks   | 39,020 |

---

## 🏗️ Project Architecture

```text
Legal Contracts
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Clause Extraction
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Clause Classification Model
        │
        ├──────────────► Risk Scoring Engine
        │
        ▼
Sentence Transformers Embeddings
        │
        ▼
FAISS Vector Database
        │
        ▼
Contract Similarity Search
        │
        ▼
RAG Pipeline
        │
        ▼
Groq LLM
        │
        ▼
Legal AI Assistant
```

---

## 🛠️ Tech Stack

### Machine Learning

* Scikit-Learn
* TF-IDF Vectorization
* Logistic Regression

### NLP

* Sentence Transformers
* Transformers

### Vector Search

* FAISS

### LLM

* Groq
* LangChain

### Frontend

* Streamlit

### Backend

* Python

### Deployment

* Docker

---

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/legal-contract-intelligence-platform.git

cd legal-contract-intelligence-platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

Application will start at:

```text
http://localhost:8501
```

---

## 🐳 Docker Deployment

Build Docker Image:

```bash
docker build -t legal-ai .
```

Run Container:

```bash
docker run -p 8501:8501 legal-ai
```

Access application:

```text
http://localhost:8501
```

---

## 📈 Machine Learning Performance

### Clause Classification Model

| Metric            | Value |
| ----------------- | ----- |
| Training Accuracy | 83.2% |
| Test Accuracy     | 69.7% |

Model Type:

```text
TF-IDF + Logistic Regression
```

---

## 🔮 Future Improvements

* Named Entity Recognition (NER)
* Clause Recommendation Engine
* Multi-Language Contract Analysis
* Legal Summarization
* Contract Comparison Dashboard
* Advanced Risk Prediction Models
* Production API Deployment

---

## 📸 Application Modules

### Dashboard

* Platform Overview
* Dataset Statistics
* Project Metrics

### Clause Classification

* Predict Clause Category
* Confidence Score

### Risk Analysis

* Risk Scoring
* Risk Categorization

### Similar Contract Search

* Semantic Contract Retrieval
* Similarity Percentage

### Legal AI Assistant

* Contract Question Answering
* RAG-Based Responses

---

## 🎯 Key Learning Outcomes

This project demonstrates:

* Machine Learning for Text Classification
* NLP and Text Processing
* Semantic Search Systems
* Vector Databases
* Retrieval-Augmented Generation (RAG)
* LLM Integration
* Streamlit Deployment
* Docker Containerization

---

## 👨‍💻 Author

**Prajwal Poojary**

Aspiring Data Scientist | Machine Learning Engineer | Generative AI Enthusiast

Connect with me on LinkedIn and GitHub.

---

## ⭐ If you found this project useful

Please consider giving the repository a star.
