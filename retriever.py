from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from eval_logger import log_query   # evaluation logging

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# SOURCE WEIGHTS (tunable)
# -----------------------------
SOURCE_WEIGHTS = {
    "faq": 1.25,        # highest precision (Q/A direct facts)
    "handbook": 1.0,    # core knowledge base
    "cases": 0.85       # slightly lower priority (clinical noise)
}


def build_retriever(
    k_faq: int = 5,
    k_cases: int = 5,
    k_handbook: int = 6,
) -> RunnableLambda:

    # -----------------------------
    # EMBEDDINGS
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # -----------------------------
    # VECTOR STORES
    # -----------------------------
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

    # -----------------------------
    # RETRIEVERS (MMR)
    # -----------------------------
    faq_retriever = faq_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k_faq,
            "fetch_k": 15,
            "lambda_mult": 0.7
        },
    )

    cases_retriever = cases_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k_cases,
            "fetch_k": 15,
            "lambda_mult": 0.7
        },
    )

    handbook_retriever = handbook_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k_handbook,
            "fetch_k": 20,
            "lambda_mult": 0.7
        },
    )

    # -----------------------------
    # SOURCE WEIGHT FUNCTION
    # -----------------------------
    def get_weight(doc: Document) -> float:
        return SOURCE_WEIGHTS.get(
            doc.metadata.get("source", "handbook"),
            1.0
        )

    # -----------------------------
    # MAIN RETRIEVAL FUNCTION
    # -----------------------------
    def retrieve(query: str) -> list[Document]:

        # --------------------------------
        # STEP 1: RAW RETRIEVAL
        # --------------------------------
        faq_docs = faq_retriever.invoke(query)
        case_docs = cases_retriever.invoke(query)
        handbook_docs = handbook_retriever.invoke(query)

        raw_docs = faq_docs + case_docs + handbook_docs

        # --------------------------------
        # STEP 2: DEDUPLICATION
        # --------------------------------
        seen = set()
        deduped_docs = []

        for doc in raw_docs:
            key = doc.page_content

            if key in seen:
                continue

            seen.add(key)
            deduped_docs.append(doc)

        # --------------------------------
        # STEP 3: SOURCE WEIGHTING
        # --------------------------------
        scored_docs = []

        for doc in deduped_docs:
            weight = get_weight(doc)

            scored_docs.append((weight, doc))

        # --------------------------------
        # STEP 4: SORT BY WEIGHT
        # --------------------------------
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        final_docs = [doc for _, doc in scored_docs]

        # --------------------------------
        # DEBUG OUTPUT (FULL TRACE)
        # --------------------------------
        print("\n" + "=" * 50)
        print("WEIGHTED RETRIEVAL PIPELINE")
        print("=" * 50)

        for i, (score, doc) in enumerate(scored_docs, start=1):

            print(f"\nDOC {i}")
            print(f"SCORE: {score}")
            print("SOURCE:", doc.metadata.get("source"))
            print("CHAPTER:", doc.metadata.get("chapter"))
            print("FAQ_ID:", doc.metadata.get("faq_id"))
            print("CASE_ID:", doc.metadata.get("case_id"))
            print("-" * 40)
            print(doc.page_content[:250])

        print("\nFINAL DOC COUNT:", len(final_docs))

        # --------------------------------
        # EVALUATION LOGGER HOOK (FIXED)
        # --------------------------------
        try:
            log_query(query, final_docs)
        except Exception as e:
            print("⚠️ Eval logging failed:", str(e))

        return final_docs

    return RunnableLambda(retrieve)