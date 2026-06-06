"""
Ingests data/medical_handbook.pdf into the 'handbook' Chroma collection.

The medical handbook is loaded, cleaned, split chapter-wise, embedded using
HuggingFace embeddings, and stored in ChromaDB.

Run:
    python ingest_pdf.py
"""

import os
import re

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

CHROMA_DIR = "chroma_store"
COLLECTION = "handbook"

PDF_PATH = os.path.join(
    "data",
    "medical_handbook.pdf"
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200


# ---------------------------------------------------
# CHAPTER SPLITTING
# ---------------------------------------------------

def split_by_chapters(pages):
    full_text = "\n".join(page.page_content for page in pages)

    chapter_pattern = re.compile(
        r"(?<!\S)(\d+\.\s+[A-Z][A-Z\s\(\)\-]{8,80})(?=\s+Definition)"
    )

    matches = list(chapter_pattern.finditer(full_text))

    if not matches:
        print("No chapter headings found. Using full document.")
        return [
            Document(
                page_content=full_text,
                metadata={
                    "chapter": "FULL_DOCUMENT",
                    "source": "handbook",
                    "source_type": "medical_handbook"
                }
            )
        ]

    chapter_docs = []

    for i, match in enumerate(matches):
        start = match.start()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(full_text)
        )

        chapter_title = match.group(1).strip()
        chapter_title = re.sub(r"\s+[A-Z]$", "", chapter_title)

        chapter_text = full_text[start:end]

        chapter_docs.append(
            Document(
                page_content=
                    f"CHAPTER: {chapter_title}\n\n"
                    + chapter_text,
                metadata={
                    "chapter": chapter_title,
                    "source": "handbook",
                    "source_type": "medical_handbook"
                }
            )
        )

    print(f"Detected {len(chapter_docs)} chapters.")

    print("\n===== CHAPTERS DETECTED =====")
    for doc in chapter_docs:
        print(doc.metadata["chapter"])

    return chapter_docs


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    print("Loading medical handbook...")

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    print(f"{len(pages)} handbook pages loaded.")

    # ---------------------------------------------------
    # CLEAN TEXT
    # ---------------------------------------------------
    print("Cleaning PDF text...")

    for page in pages:
        text = page.page_content

        # Remove Gemini URLs
        text = re.sub(
            r"https?://\S+",
            " ",
            text
        )

        text = re.sub(
            r"gemini\.google\.com\S*",
            " ",
            text,
            flags=re.IGNORECASE
        )

        # Remove page markers
        text = re.sub(
            r"\b\d+\s*/\s*\d+\b",
            " ",
            text
        )

        # Remove Gemini artifacts
        text = re.sub(r".*Google Gemini.*", "", text, flags=re.IGNORECASE)

        # Remove timestamps
        text = re.sub(
            r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(AM|PM)",
            "",
            text,
            flags=re.IGNORECASE
        )

        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Remove standalone page numbers
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # Normalize spaces
        text = re.sub(r"\s+", " ", text)

        page.page_content = text.strip()

    # Remove tiny garbage pages
    pages = [page for page in pages if len(page.page_content.strip()) > 50]

    print(f"{len(pages)} cleaned pages retained.")

    # ---------------------------------------------------
    # CHAPTER SPLITTING
    # ---------------------------------------------------
    chapter_docs = split_by_chapters(pages)

    print("\n===== CHAPTERS DETECTED =====")
    for doc in chapter_docs:
        print("\n------------------")
        print("CHAPTER:")
        print(doc.metadata["chapter"])
        print("\nSTARTS WITH:")
        print(doc.page_content[:300])

    # ---------------------------------------------------
    # CHUNKING
    # ---------------------------------------------------
    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(chapter_docs)

    chunks = [
        chunk
        for chunk in chunks
        if len(chunk.page_content.strip()) > 100
    ]

    print("\n===== FIRST 5 CHUNKS =====")
    for i, chunk in enumerate(chunks[:5]):
        print("\n------------------")
        print(f"CHUNK {i}")
        print("CHAPTER:")
        print(chunk.metadata.get("chapter"))
        print("\nCONTENT:")
        print(chunk.page_content[:500])

    # ---------------------------------------------------
    # METADATA ENRICHMENT
    # ---------------------------------------------------
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "handbook"
        chunk.metadata["source_type"] = "medical_handbook"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chapter"] = chunk.metadata.get("chapter", "Unknown")

    print(f"{len(chunks)} chunks produced.")

    # ---------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------
    print("Initialising embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    if not chunks:
        raise ValueError("No chunks generated from PDF.")

    # ---------------------------------------------------
    # STORE IN CHROMA
    # ---------------------------------------------------
    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    print(f"Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()