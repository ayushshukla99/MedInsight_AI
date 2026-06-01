from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


def rewrite_query(query, memory):

    history = memory.get_history()

    if not history:
        return query

    prompt = ChatPromptTemplate.from_template(
        """
You are a medical query rewriting assistant.

Conversation History:
{history}

Latest User Question:
{query}

Rewrite the latest question into a complete standalone question.

Rules:
- Resolve pronouns.
- Resolve references like:
  it, its, this, that, they.
- Do not answer.
- Only return the rewritten question.
"""
    )

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    rewritten = chain.invoke(
        {
            "history": history,
            "query": query
        }
    )
    if "</think>" in rewritten:
        rewritten = rewritten.split("</think>")[-1].strip()

    return rewritten.strip()