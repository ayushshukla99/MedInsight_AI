# eval_logger.py

from collections import defaultdict
from datetime import datetime

# In-memory log (simple + reliable for Streamlit dev)
LOGS = []


def log_query(query: str, retrieved_docs: list):
    """
    Stores query + retrieved documents for evaluation.
    """

    LOGS.append({
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "docs": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "chapter": doc.metadata.get("chapter"),
                "faq_id": doc.metadata.get("faq_id"),
                "case_id": doc.metadata.get("case_id"),
            }
            for doc in retrieved_docs
        ]
    })


def get_logs():
    return LOGS


def clear_logs():
    LOGS.clear()