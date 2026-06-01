"""
Ingests data/medical_handbook.pdf into the 'handbook' Chroma collection.

The medical handbook is loaded, split into smaller chunks,
embedded using HuggingFace embeddings, and stored in ChromaDB.

This allows the RAG system to retrieve relevant medical
knowledge from textbook content during question answering.

Run:
    python ingest_pdf.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "handbook"
PDF_PATH = os.path.join("data", "medical_handbook.pdf")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE    = 700
CHUNK_OVERLAP = 150


def main():
    print("Loading medical handbook...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"  {len(pages)} handbook pages loaded.")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE ,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    chunks = splitter.split_documents(pages)

    # Add metadata to each chunk for citation generation
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "handbook"
        chunk.metadata["chunk_index"] = i

        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    print(f"  {len(chunks)} chunks produced.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if not chunks or len(chunks) == 0:
        raise ValueError("No chunks generated from PDF. Check extraction step.")

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()
