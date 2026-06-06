import json
from collections import defaultdict

# import your retriever + reranker pipeline
from your_retriever_file import retrieve_documents
from your_reranker_file import rerank_documents


# -----------------------------
# METRICS STORAGE
# -----------------------------

metrics = {
    "recall_at_3": [],
    "mrr": [],
    "source_accuracy": []
}


# -----------------------------
# METRIC FUNCTIONS
# -----------------------------

def compute_recall_at_k(retrieved_sources, expected_sources):
    """
    1 if ANY expected source is in top-k results
    """
    return int(len(set(retrieved_sources) & set(expected_sources)) > 0)


def compute_mrr(retrieved_sources, expected_sources):
    """
    Mean Reciprocal Rank
    """
    for i, src in enumerate(retrieved_sources):
        if src in expected_sources:
            return 1 / (i + 1)
    return 0


def extract_sources(docs):
    return [doc.metadata.get("source") for doc in docs]


# -----------------------------
# MAIN EVALUATION LOOP
# -----------------------------

def evaluate():
    with open("eval_dataset.json", "r") as f:
        dataset = json.load(f)

    for item in dataset:
        query = item["question"]
        expected = item["expected_sources"]

        print("\n====================")
        print("QUERY:", query)

        # STEP 1: retrieve
        docs = retrieve_documents(query)

        # STEP 2: rerank
        docs = rerank_documents(query, docs)

        retrieved_sources = extract_sources(docs[:3])

        print("Retrieved:", retrieved_sources)
        print("Expected:", expected)

        # METRICS
        recall = compute_recall_at_k(retrieved_sources, expected)
        mrr = compute_mrr(retrieved_sources, expected)

        metrics["recall_at_3"].append(recall)
        metrics["mrr"].append(mrr)

    # -----------------------------
    # FINAL RESULTS
    # -----------------------------

    final_recall = sum(metrics["recall_at_3"]) / len(metrics["recall_at_3"])
    final_mrr = sum(metrics["mrr"]) / len(metrics["mrr"])

    print("\n====================")
    print("FINAL RESULTS")
    print("Recall@3:", round(final_recall, 3))
    print("MRR:", round(final_mrr, 3))


if __name__ == "__main__":
    evaluate()