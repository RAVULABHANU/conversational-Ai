
# Conversational AI Agent — Upgraded (LangChain + LangGraph + RAGs + Multi-Agent)

**Goal:** Professional, production-ready upgrade of the original RAG app that attracts recruiter / professional attention.

Features
- LangChain-based Retrieval-Augmented Generation (RAG) using FAISS.
- LangGraph workflow with modular nodes (DocumentLoader, EmbeddingNode, RetrievalNode, LLMNode).
- Multi-agent orchestration: RetrieverAgent, SummarizerAgent, AnswerAgent coordinated by an Orchestrator.
- Streamlit frontend with sidebar toggle: "Standard RAG" vs "Multi-Agent (LangGraph)"
- Flexible LLM backend: OpenAI (default) with optional Hugging Face support.
- Environment-driven configuration, sample `.env.sample` included.
- Dockerfile & CI workflow template (GitHub Actions) included for professional delivery.
