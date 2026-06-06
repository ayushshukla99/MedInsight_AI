"""
Ingests clinical cases from data/clinical_cases.json
into the 'cases' Chroma collection.

Run:
    python ingest_cases.py

Each clinical case is converted into a LangChain Document,
embedded using HuggingFace embeddings, and stored inside ChromaDB.
"""

import os
import json

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

CHROMA_DIR = "chroma_store"

COLLECTION = "cases"

JSON_PATH = os.path.join(
    "data",
    "clinical_cases.json"
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------------------------------------
# Load Clinical Cases
# ------------------------------------------------------------------

def load_case_documents(json_path: str) -> list[Document]:
    """
    Reads clinical_cases.json and converts each case
    into a LangChain Document.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    docs = []

    for case in cases:

        content = (
            f"Clinical Case\n"
            f"Specialty: {case['specialty']}\n"
            f"Symptoms: {case['symptoms']}\n"
            f"History: {case['history']}\n"
            f"Investigations: {case['investigations']}\n"
            f"Diagnosis: {case['diagnosis']}\n"
            f"Treatment: {case['treatment']}\n"
            f"Outcome: {case['outcome']}"
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "cases",
                    "case_id": str(case["case_id"]),
                    "specialty": case["specialty"],
                    "diagnosis": case["diagnosis"],
                    "source_type": "clinical_case"
                }
            )
        )

    return docs


# ------------------------------------------------------------------
# Main Ingestion Pipeline
# ------------------------------------------------------------------

def main():
    """
    Loads clinical cases, creates embeddings,
    and stores them inside ChromaDB.
    """

    print("Loading clinical cases...")

    docs = load_case_documents(JSON_PATH)

    print(f"  {len(docs)} clinical cases loaded.")

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