
# A simple multi-agent orchestrator that runs three agents and merges outputs.
from agents import RetrieverAgent, SummarizerAgent, AnswerAgent

class Orchestrator:
    def __init__(self, persist_dir='./faiss_index', use_hf=False):
        self.retriever = RetrieverAgent(persist_dir=persist_dir, use_hf=use_hf)
        self.summarizer = SummarizerAgent()
        self.answerer = AnswerAgent()

    def run(self, query, uploaded=None):
        # Step 1: retrieve
        docs = self.retriever.retrieve(query, uploaded=uploaded)
        # Step 2: summarize retrieved docs
        summary = self.summarizer.summarize(docs, query)
        # Step 3: answer using both retrieved docs and summary
        final = self.answerer.answer(query, docs, summary)
        return {
            "agent_outputs": {
                "retriever": docs[:3] if isinstance(docs, list) else docs,
                "summarizer": summary,
                "answerer": final
            },
            "final_answer": final
        }
