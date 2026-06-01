"""
Builds a merged retriever across all three Chroma collections:

- faq       : Medical FAQ entries
- cases     : Clinical case studies
- handbook  : Medical handbook chunks
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR = "chroma_store"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever(
    k_faq: int = 4,
    k_cases: int = 4,
    k_handbook: int = 8,
) -> RunnableLambda:

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    cases_store = Chroma(
        collection_name="cases",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    handbook_store = Chroma(
        collection_name="handbook",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    faq_retriever = faq_store.as_retriever(
        search_kwargs={"k": k_faq}
    )

    cases_retriever = cases_store.as_retriever(
        search_kwargs={"k": k_cases}
    )

    handbook_retriever = handbook_store.as_retriever(
        search_kwargs={"k": k_handbook}
    )

    def retrieve(query: str) -> list[Document]:

        faq_docs = faq_retriever.invoke(query)

        case_docs = cases_retriever.invoke(query)

        handbook_docs = handbook_retriever.invoke(query)

        all_docs = (
            faq_docs
            + case_docs
            + handbook_docs
        )

        seen = set()
        unique_docs = []

        for doc in all_docs:

            if doc.page_content not in seen:

                seen.add(doc.page_content)

                unique_docs.append(doc)

        return unique_docs

    return RunnableLambda(retrieve)