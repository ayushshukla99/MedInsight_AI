# eval_metrics.py

from collections import Counter
from eval_logger import get_logs


def _extract_relevant_source(query: str):
    """
    Simple heuristic ground-truth mapping.
    You can later replace with labeled dataset.
    """

    q = query.lower()

    if "symptom" in q or "treatment" in q:
        return "handbook"
    if "case" in q or "patient" in q:
        return "cases"
    return "faq"


def evaluate(k: int = 3):
    """
    Computes retrieval metrics over logged queries.
    """

    logs = get_logs()

    if not logs:
        return {
            "total_queries": 0,
            "recall_at_3": 0.0,
            "mrr": 0.0,
            "source_distribution": {}
        }

    total = len(logs)

    recall_hits = 0
    mrr_total = 0.0
    source_counter = Counter()

    for entry in logs:
        query = entry["query"]
        docs = entry["docs"]

        expected_source = _extract_relevant_source(query)

        found = False
        rank_found = 0

        for i, doc in enumerate(docs[:k]):
            source_counter[doc["source"]] += 1

            if doc["source"] == expected_source and not found:
                found = True
                rank_found = i + 1

        if found:
            recall_hits += 1
            mrr_total += 1.0 / rank_found
        else:
            mrr_total += 0.0

    return {
        "total_queries": total,
        "recall_at_3": recall_hits / total,
        "mrr": mrr_total / total,
        "source_distribution": dict(source_counter)
    }