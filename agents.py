
# Simple agent implementations. Keep stateless and testable.
from utils.rag_chain import get_retriever_docs, build_retrieval_qa_chain, simple_qa
from utils.embedding_utils import load_faiss_index

class RetrieverAgent:
    def __init__(self, persist_dir='./faiss_index', use_hf=False):
        self.persist_dir = persist_dir
        self.use_hf = use_hf

    def retrieve(self, query, uploaded=None, k=5):
        # load retriever and return top k documents (as text snippets)
        retriever = load_faiss_index(self.persist_dir, use_hf=self.use_hf)
        docs = get_retriever_docs(retriever, query, k=k)
        return docs

class SummarizerAgent:
    def summarize(self, docs, query):
        # Lightweight summarization strategy (could call an LLM)
        if not docs:
            return "No documents retrieved."
        joined = "\n\n".join([d.page_content[:800] for d in docs])
        # simple heuristic: return first 800 chars
        return joined[:1500]

class AnswerAgent:
    def answer(self, query, docs, summary):
        # Use the summary + docs to produce the final answer via a QA chain
        # Fallback to summary if chain not available
        try:
            chain = build_retrieval_qa_chain(persist_dir='./faiss_index', use_hf=False)
            return simple_qa(chain, query)
        except Exception as e:
            return f"Answer (fallback): Based on summary - {summary[:800]}"
