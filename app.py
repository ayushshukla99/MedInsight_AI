import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from rag_chain import build_chain
from eval_metrics import evaluate


# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_QUESTIONS = [
    "What is Type 2 Diabetes?",
    "What are the symptoms of myocardial infarction?",
    "How is hypertension diagnosed?",
    "What is the treatment for asthma?",
    "What causes diabetic ketoacidosis?",
    "How is COPD managed?",
]

st.set_page_config(
    page_title="Medical RAG",
    page_icon="🏥",
    layout="centered",
)


# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>

    .stApp {
        background-color: #0f1117;
        color: #e6e6e6;
    }

    section[data-testid="stSidebar"] {
        background-color: #161a22;
    }

    /* FIX CHAT INPUT (BOTTOM FIXED LIKE CHATGPT) */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 20px;
        width: 50%;
        z-index: 9999;
        background: #161a22;
        padding: 10px;
        border-radius: 12px;
    }

    .block-container {
        padding-bottom: 120px;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# CACHE CHAIN
# -----------------------------
@st.cache_resource
def get_chain():
    return build_chain()


# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

if "mode" not in st.session_state:
    st.session_state.mode = "chat"


def set_mode(mode):
    st.session_state.mode = mode
    st.rerun()   # IMPORTANT FIX


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.title("🏥 Medical Assistant")
    st.caption("RAG + Evaluation System")

    st.divider()

    # -----------------------------
    # MODE SWITCH
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Chat", use_container_width=True):
            set_mode("chat")

    with col2:
        if st.button("Eval", use_container_width=True):
            set_mode("eval")

    st.divider()

    # -----------------------------
    # SAMPLE QUESTIONS
    # -----------------------------
    st.markdown("### Sample Questions")

    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    # -----------------------------
    # CLEAR CHAT
    # -----------------------------
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []

    st.divider()

    # -----------------------------
    # RUN EVALUATION
    # -----------------------------
    if st.button("▶ Run Evaluation", use_container_width=True):

        with st.spinner("Running RAG evaluation..."):
            results = evaluate()

        st.session_state.eval_results = results
        st.rerun()   # IMPORTANT FIX


# =====================================================
# MAIN TITLE
# =====================================================
st.title("Medical Answer Verifier")
st.caption("Evidence-backed medical RAG system")


# =====================================================
# CHAT MODE
# =====================================================
if st.session_state.mode == "chat":

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a medical question...")

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            chain = get_chain()

            with st.spinner("Retrieving medical evidence..."):
                response = chain(question)

            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
        })


# =====================================================
# EVALUATION MODE
# =====================================================
else:

    st.subheader("📊 RAG Evaluation Dashboard")

    # IMPORTANT FIX: None check (not truthy check)
    if st.session_state.eval_results is not None:

        results = st.session_state.eval_results

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Queries", results.get("total_queries", 0))
        col2.metric("Recall@3", round(results.get("recall_at_3", 0.0), 3))
        col3.metric("MRR", round(results.get("mrr", 0.0), 3))

        st.divider()

        st.subheader("Source Distribution")
        st.json(results.get("source_distribution", {}))

        # DEBUG (VERY USEFUL)
        st.divider()
        st.subheader("Raw Output Debug")
        st.json(results)

    else:
        st.info("Run evaluation from sidebar to generate metrics.")