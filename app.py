
import os
import streamlit as st
from utils.embedding_utils import get_embeddings, build_faiss_index, load_faiss_index
from utils.rag_chain import build_retrieval_qa_chain, simple_qa
from langgraph_workflow import run_langgraph_workflow
from orchestrator import Orchestrator
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Conversational AI Agent — Upgraded", layout="wide")
st.title("Conversational AI Agent — LangChain + LangGraph + Multi-Agent")

st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("Mode", ["rag", "multi_agent"], index=1 if os.getenv('DEFAULT_MODE','multi_agent')=='multi_agent' else 0)
use_hf = st.sidebar.checkbox("Use HuggingFace embeddings (if configured)", value=False)
persist_dir = st.sidebar.text_input("FAISS persist dir", value=os.getenv("PERSIST_DIR","./faiss_index"))

uploaded = st.file_uploader("Upload documents (PDF / TXT). You can upload multiple.", type=['pdf','txt'], accept_multiple_files=True)
if uploaded:
    st.sidebar.success(f"{len(uploaded)} file(s) uploaded")
    # Build index (simple flow)
    with st.spinner("Building / updating FAISS index..."):
        index = build_faiss_index(uploaded, persist_dir, use_hf)
    st.success("Index ready.")
else:
    index = None

query = st.text_input("Ask a question about your documents:")
if st.button("Get Answer") and query:
    if mode == "rag":
        if index is None:
            st.error("Please upload documents to build the FAISS index first.")
        else:
            chain = build_retrieval_qa_chain(persist_dir, use_hf=use_hf)
            answer = simple_qa(chain, query)
            st.subheader("Answer (RAG)")
            st.write(answer)
    else:
        # multi-agent LangGraph workflow
        orchestrator = Orchestrator(persist_dir=persist_dir, use_hf=use_hf)
        with st.spinner("Running multi-agent workflow..."):
            result = orchestrator.run(query, uploaded=uploaded)
        st.subheader("Answer (Multi-Agent)")
        st.write(result['final_answer'])
        st.markdown("**Agent Outputs:**")
        for k,v in result.get('agent_outputs', {}).items():
            st.markdown(f"**{k}**: {v}")
