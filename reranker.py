from sentence_transformers import CrossEncoder

# Load once at startup
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank_documents(
    query: str,
    docs: list,
    top_k: int = 3
):
    """
    Rerank retrieved documents using BGE CrossEncoder.

    Parameters
    ----------
    query : str
        User query after rewriting.

    docs : list
        Retrieved LangChain Document objects.

    top_k : int
        Number of final documents to keep.

    Returns
    -------
    list
        Top reranked documents.
    """

    if not docs:
        return []

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(
        zip(scores, docs)
    )

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print("\n===== RERANKED SCORES =====")

    for rank, (score, doc) in enumerate(scored_docs, start=1):
        print(f"\nRANK {rank}")
        print(f"SCORE: {float(score):.4f}")
        print(doc.page_content[:200])

    # Keep only the strongest candidates
    best_score = float(scored_docs[0][0])

    filtered_docs = [
        doc
        for score, doc in scored_docs
        if float(score) >= best_score * 0.5
    ]

    # If filtering removes everything,
    # fallback to top_k highest scoring docs
    if len(filtered_docs) == 0:
        filtered_docs = [
            doc
            for score, doc in scored_docs[:top_k]
        ]

    return filtered_docs[:top_k]