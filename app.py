import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from rag_chain import build_chain



SAMPLE_QUESTIONS = [
    "What is Type 2 Diabetes?",
    "What are the symptoms of myocardial infarction?",
    "How is hypertension diagnosed?",
    "What is the treatment for asthma?",
    "What causes diabetic ketoacidosis?",
    "How is COPD managed?",
]

st.set_page_config(
    page_title="medical answer",
    page_icon="+",
    layout="centered",
)

@st.cache_resource
def get_chain():
    return build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# Sidebar
with st.sidebar:

    st.title("🏥 Medical Assistant")
    st.caption("Powered by RAG + Citations")

    st.divider()

    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")

    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []


# Main
st.title("Medical Answer Verifier")
st.caption("Ask medical questions and receive evidence-backed answers.")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.chat_input("Describe your issue…")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        chain = get_chain()

        with st.spinner("Searching medical sources..."):
            response = chain(question)

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )