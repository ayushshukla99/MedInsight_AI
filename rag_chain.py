"""
Medical RAG Chain v2
Memory + Query Rewrite + Multi-Collection Retrieval
"""

from memory import SessionMemory
from rewrite import rewrite_query

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever
from reranker import rerank_documents

# ------------------------------------------------------------------
# GLOBAL OBJECTS
# ------------------------------------------------------------------

memory = SessionMemory()

RETRIEVER = build_retriever()

LLM = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


# ------------------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------------------

SYSTEM_PROMPT = """
You are MedVerify AI, an evidence-based medical knowledge assistant.

You MUST answer ONLY using the provided context.

Sources:
1. FAQs
2. Clinical Cases
3. Medical Handbook

Rules:

- Do NOT hallucinate.
- Do NOT guess.
- If context is insufficient, say:

"The available medical sources do not contain enough information to answer confidently."

- When answering:

- Prefer FAQ answers for direct factual questions.
- Prefer Clinical Cases for diagnosis/treatment examples.
- Prefer Handbook content for detailed medical explanations.
- If multiple sources agree, synthesize them.
- Never use knowledge outside the provided context.
- Be medically accurate.
- Prefer retrieved evidence over assumptions.

Context:
{context}
"""


# ------------------------------------------------------------------
# FORMAT DOCS
# ------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:

    sections = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "unknown"
        ).upper()

        sections.append(
            f"[SOURCE: {source}]\n{doc.page_content}"
        )

    return "\n\n------------------------------\n\n".join(
        sections
    )


# ------------------------------------------------------------------
# MAIN RAG FUNCTION
# ------------------------------------------------------------------

def run_rag(user_question: str):

    # --------------------------------------
    # Rewrite Query
    # --------------------------------------

    rewritten_query = rewrite_query(
        user_question,
        memory
    )

    print("\n========================")
    print("ORIGINAL QUERY:")
    print(user_question)

    print("\nREWRITTEN QUERY:")
    print(rewritten_query)
    print("========================")

    # --------------------------------------
    # Retrieve Documents
    # --------------------------------------

    docs = RETRIEVER.invoke(
        rewritten_query
    )
    docs = rerank_documents(
        rewritten_query,
        docs,
        top_k=5
    )

    print("\n===== RERANKED DOCS =====")

    for i, doc in enumerate(docs, start=1):
        print(f"\nDOC {i}")
        print(doc.page_content[:300])

    if not docs:
        return "No relevant medical context found."

    print("\nTOP RETRIEVED DOCUMENT:")
    print(docs[0].page_content[:300])

    print("\n===== RETRIEVED DOCS =====")

    for i, doc in enumerate(docs, start=1):
        print(f"\nDOC {i}")
        print(doc.page_content[:300])

    # --------------------------------------
    # Build Context
    # --------------------------------------

    context = _format_docs(docs)

    # --------------------------------------
    # Prompt
    # --------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ]
    )

    chain = (
        prompt
        | LLM
        | StrOutputParser()
    )

    # --------------------------------------
    # Generate Response
    # --------------------------------------

    response = chain.invoke(
        {
            "context": context,
            "question": user_question
        }
    )

    # --------------------------------------
    # Update Memory
    # --------------------------------------

    memory.add_turn(
    user_question,
    response
)

    return response


# ------------------------------------------------------------------
# APP COMPATIBILITY
# ------------------------------------------------------------------

def build_chain():
    return run_rag