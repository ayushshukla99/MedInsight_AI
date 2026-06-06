from sentence_transformers import CrossEncoder

# Load once at startup
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

# -----------------------------------
# SOURCE WEIGHT CONFIG
# -----------------------------------
SOURCE_WEIGHTS = {
    "handbook": 1.0,
    "cases": 0.85,
    "faq": 0.70
}


def get_source_weight(source: str) -> float:
    """
    Returns weight for document source.
    Default = 0.6 for unknown sources.
    """
    return SOURCE_WEIGHTS.get(source, 0.6)


def rerank_documents(
    query: str,
    docs: list,
    top_k: int = 3
):
    """
    Rerank retrieved documents using:
    1. CrossEncoder semantic score
    2. Source-aware weighting
    """

    if not docs:
        return []

    # ----------------------------------
    # Build query-document pairs
    # ----------------------------------
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    # Cross-encoder scores
    scores = reranker.predict(pairs)

    scored_docs = []

    # ----------------------------------
    # Combine semantic + source weight
    # ----------------------------------
    for score, doc in zip(scores, docs):

        score = float(score)

        source = doc.metadata.get("source", "unknown")
        weight = get_source_weight(source)

        # FINAL SCORE = semantic × source weight
        final_score = score * weight

        doc.metadata["rerank_score"] = final_score
        doc.metadata["semantic_score"] = score
        doc.metadata["source_weight"] = weight

        scored_docs.append(
            (final_score, doc)
        )

    # Sort by final score
    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print("\n===== RERANKED SCORES (WEIGHTED) =====")

    for rank, (score, doc) in enumerate(
        scored_docs,
        start=1
    ):
        print(f"\nRANK {rank}")
        print(f"FINAL SCORE: {score:.4f}")
        print(f"SEMANTIC SCORE: {doc.metadata.get('semantic_score'):.4f}")
        print(f"SOURCE WEIGHT: {doc.metadata.get('source_weight'):.2f}")
        print("SOURCE:", doc.metadata.get("source"))
        print("CHAPTER:", doc.metadata.get("chapter"))
        print(doc.page_content[:200])

    # ----------------------------------
    # Adaptive thresholding
    # ----------------------------------
    best_score = scored_docs[0][0]

    threshold = max(
        best_score * 0.55,  # slightly stricter than before
        0.30
    )

    print(f"\nTHRESHOLD: {threshold:.4f}")

    filtered_docs = [
        doc
        for score, doc in scored_docs
        if score >= threshold
    ]

    # fallback safety
    if not filtered_docs:
        filtered_docs = [
            doc
            for score, doc in scored_docs[:top_k]
        ]

    print("\n===== AFTER THRESHOLD =====")

    for doc in filtered_docs:
        print(f"SOURCE: {doc.metadata.get('source')}")
        print(f"CHAPTER: {doc.metadata.get('chapter')}")
        print(doc.page_content[:150])
        print("-" * 50)

    print("\nDocuments kept:", len(filtered_docs))

    return filtered_docs[:top_k]