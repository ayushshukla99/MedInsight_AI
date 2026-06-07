import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("evaluation/eval_log.json")


def log_query(query: str, retrieved_docs: list):
    entry = {
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
    }

    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    else:
        logs = []

    logs.append(entry)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)