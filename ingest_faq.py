"""
Ingests data/medical_faq.csv into the 'faq' Chroma collection.

Each FAQ entry is converted into a LangChain Document,
embedded using HuggingFace embeddings, and stored in ChromaDB.

These FAQs serve as one of the knowledge sources used by the
Medical Answer Verifier RAG system.

Run:
    python ingest_faq.py
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import pandas as pd

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

CHROMA_DIR = "chroma_store"

COLLECTION = "faq"

CSV_PATH = os.path.join(
    "data",
    "medical_faq.csv"
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------------
# Load FAQ Documents
# ------------------------------------------------------------------

def load_faq_documents(csv_path: str) -> list[Document]:
    """
    Reads the medical FAQ CSV file and converts each row
    into a LangChain Document.

    Returns:
        list[Document]
    """

    df = pd.read_csv(csv_path)

    docs = []

    for _, row in df.iterrows():

        content = (
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}"
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "faq",
                    "faq_id": str(row["id"]),
                    "category": row["category"],
                    "question": row["question"],
                },
            )
        )

    return docs


# ------------------------------------------------------------------
# Main Ingestion Pipeline
# ------------------------------------------------------------------

def main():
    """
    Loads medical FAQs, generates embeddings,
    and stores them inside ChromaDB.
    """

    print("Loading medical FAQs...")

    docs = load_faq_documents(CSV_PATH)

    print(f"  {len(docs)} FAQ entries loaded.")

    print("Initialising embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    print(
        f"Embedding and storing in Chroma collection '{COLLECTION}'..."
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print(
        f"  Done. {vectorstore._collection.count()} vectors stored."
    )


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()