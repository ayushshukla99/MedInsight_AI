🧠 Medical RAG Chat Assistant
⚡ Overview

A Retrieval-Augmented Generation (RAG) based medical chatbot that answers health-related queries using trusted medical sources stored in a structured vector database.

It combines:

📚 Medical FAQ knowledge base
🧾 Clinical case studies
📖 Medical handbook content
🧠 Semantic retrieval + reranking
🤖 LLM reasoning (Groq Qwen3-32B)

The system is designed to reduce hallucinations and improve factual grounding in medical Q&A.

🏗️ Architecture
User Query
   ↓
Query Rewriting (optional normalization)
   ↓
Merged Retriever (Chroma DB)
   ├── FAQ Collection
   ├── Case Studies Collection
   └── Handbook Collection
   ↓
Initial Semantic Retrieval (Embeddings)
   ↓
Cross-Encoder Reranking (Relevance boost)
   ↓
Top-K Context Selection
   ↓
Prompt Construction
   ↓
Groq LLM (Qwen3-32B)
   ↓
Final Answer
🧠 Core Components
1. Vector Database (Chroma)

Stores embeddings of:

Medical FAQs
Clinical case studies
Medical handbook chunks

Each document is embedded using HuggingFace sentence transformers.

2. Merged Retriever

Custom retriever that:

Queries all 3 collections
Combines results
Normalizes scores
Returns unified ranked context
3. Cross-Encoder Reranking

Improves retrieval precision by:

Re-scoring retrieved chunks
Prioritizing semantically stronger matches
Filtering noise
4. Query Rewriting Layer

Enhances retrieval by:

Rephrasing vague user queries
Expanding medical terminology
Improving embedding alignment
5. LLM Layer (Groq)
Model: Qwen3-32B
Role: Final reasoning + response generation
Input: structured context + user query
⚙️ Tech Stack
LangChain – orchestration
ChromaDB – vector storage
HuggingFace Transformers – embeddings
CrossEncoder (SentenceTransformers) – reranking
Groq API – LLM inference
Python – backend logic
🚀 Features
🧠 Medical domain-focused RAG pipeline
📚 Multi-source knowledge retrieval
🎯 Reranking for higher accuracy answers
🔁 Query rewriting for better recall
⚡ Fast inference via Groq
🧾 Structured and explainable responses
📦 Project Structure
Medical Assistant/
│
├── retriever.py        # Merged retriever logic (FAQ + cases + handbook)
├── ingest.py           # Embedding + Chroma DB setup
├── rag_chain.py        # Prompt + LLM pipeline
├── query_rewrite.py    # Query enhancement layer
├── rerank.py           # Cross-encoder reranking logic
├── chroma_store/       # Vector DB storage
└── main.py             # Entry point (chat interface)
🧪 How It Works (Step-by-Step)
User asks a medical question
Query is optionally rewritten for clarity
Retriever searches all 3 vector stores
Top documents are ranked using Cross-Encoder
Best chunks are selected as context
Prompt is constructed with strict medical grounding
Groq LLM generates final answer
Response returned to user
📌 Example Query

User:

What are the symptoms of appendicitis?

System Flow:

Retrieves case studies of abdominal pain
Pulls FAQ on appendicitis
Matches handbook definition
Reranks relevant symptoms context
LLM generates structured answer
⚠️ Disclaimer

This system is for educational and informational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.

🔥 Future Improvements
 Hybrid search (BM25 + vector)
 Medical entity extraction layer
 Confidence scoring for answers
 Citation-based response output
 UI chatbot interface (Streamlit / React)
 Feedback loop for retraining retrieval quality
💡 Why This Project Matters

Most LLM medical bots fail due to hallucination.
This system solves that using:

grounded retrieval
reranking
structured knowledge separation

Result: more accurate + explainable medical answers