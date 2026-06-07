🧠 MedInsight_AI
Evidence-Backed Medical RAG System

A Retrieval-Augmented Generation (RAG) system for medical question answering using verified clinical sources with evaluation-driven reliability (MRR, Recall@K, source tracking).

🚀 Overview

MedInsight_AI is a domain-specific medical QA system built using a RAG architecture.

It retrieves relevant medical knowledge from structured sources and generates grounded, evidence-based answers with minimal hallucination.

The system includes a full evaluation pipeline to measure retrieval quality and system reliability.

⚙️ Key Features
🔎 Semantic search using embeddings
📚 Multi-source knowledge base:
Medical FAQs
Clinical Handbook (PDF)
Case-based reasoning data
🧠 Context-aware LLM response generation
📊 Evaluation system:
Recall@K
Mean Reciprocal Rank (MRR)
Source distribution analysis
🧾 Structured logging of all evaluation runs
🧪 Custom medical benchmark dataset
🏗️ System Architecture
User Query
   ↓
Embedding Model
   ↓
Vector Database (ChromaDB)
   ↓
Top-K Retrieval (FAQ + Handbook + Cases)
   ↓
Context Ranking / Filtering
   ↓
LLM Answer Generation
   ↓
Final Evidence-Based Medical Answer
📂 Project Structure
MedInsight_AI/
│
├── app.py                         # Main RAG application
│
├── evaluation/
│   ├── eval_metrics.py            # MRR & Recall calculation
│   ├── eval_logger.py             # Logging evaluation runs
│   ├── eval_dataset.json          # Evaluation questions
│   ├── eval_log.json              # Raw retrieval outputs
│   ├── eval_report.md             # Final evaluation report
│
├── data/
│   ├── faq/
│   ├── handbook/
│   ├── cases/
│
├── retriever/
├── embeddings/
├── vector_store/
└── utils/
📊 Evaluation Results
Metric	Score
Recall@3	0.80
MRR	0.80
📦 Source Distribution
FAQ → dominant retrieval source
Handbook → moderate usage
Case data → minimal but important for reasoning
🧪 Evaluation Methodology

The system is tested on 10 curated medical queries covering:

Medical definitions
Diagnostic criteria
Treatment guidelines
Pathophysiology
Clinical reasoning
Multi-step comparisons
📸 Screenshots
1. Chat Interface (RAG Output)

Add screenshot of your chatbot answering medical queries

📍 Replace below with image once uploaded
![Chat Interface](screenshots/chat.png)
2. Evaluation Dashboard

Add screenshot of evaluation metrics (MRR, Recall, logs)

![Evaluation Dashboard](screenshots/eval.png)
📁 How to Add Screenshots
Step 1: Create folder
mkdir screenshots
Step 2: Add images

Place files like:

screenshots/chat.png
screenshots/eval.png
Step 3: Reference in README

Already done above:

![Chat Interface](screenshots/chat.png)
![Evaluation Dashboard](screenshots/eval.png)
⚠️ Limitations
Small evaluation dataset (20 queries)
FAQ-heavy retrieval bias
Limited rare disease coverage
No real-time clinical validation layer
🔮 Future Improvements
Add MMR-based diversified retrieval
Improve handbook chunking strategy
Expand dataset (100+ medical queries)
Add LLM-based faithfulness scoring
Build CI-based evaluation pipeline
Add real-time medical knowledge updates
🧠 Core Principle

“Every answer must be traceable to retrieved medical evidence.”

🛠️ Tech Stack
Python
LangChain / Custom RAG pipeline
ChromaDB (Vector DB)
Sentence Transformers
JSON-based evaluation system
📌 How to Run
Start app
python app.py
Run evaluation
python evaluation/eval_metrics.py
Generate report
python evaluation/eval_logger.py
📈 Project Status

✔ Retrieval system working
✔ Evaluation pipeline active
✔ Grounded responses enforced
⚠ Needs dataset scaling + retrieval tuning

🧾 License

For academic and portfolio use only.

⭐ Final Note

This project demonstrates:

Real-world RAG architecture
Evaluation-driven ML system design
Medical domain grounding
Production-style structuring