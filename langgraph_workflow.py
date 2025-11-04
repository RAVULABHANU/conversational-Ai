
# Concrete LangGraph workflow implemented using the local lightweight langgraph.Graph
from langgraph import Graph, Node
from utils.embedding_utils import get_embeddings, build_faiss_index, load_faiss_index, _load_texts
from utils.rag_chain import build_retrieval_qa_chain, simple_qa

def run_langgraph_workflow(query, persist_dir, uploaded_files=None, use_hf=False):
    # Build a simple graph: DocumentLoader -> Embed/Index -> Retrieve -> LLM
    g = Graph(name='rag_workflow')

    def loader_fn(inputs):
        # inputs will be {'DocumentLoader': uploaded_files} or similar
        files = inputs.get('DocumentLoader') if isinstance(inputs, dict) else uploaded_files
        return _load_texts(files) if files else []

    def index_fn(docs):
        # docs: list of langchain Document
        # Build or update FAISS index; ignore return
        if docs:
            # build_faiss_index expects file-like objects; we reuse by saving docs to temp if needed
            return build_faiss_index(uploaded_files, persist_dir, use_hf)
        return None

    def retrieve_fn(_):
        # load retriever and fetch top documents for query
        retriever = load_faiss_index(persist_dir, use_hf=use_hf).as_retriever(search_kwargs={'k':5})
        docs = retriever.get_relevant_documents(query)
        return docs

    def llm_fn(inputs):
        docs = inputs.get('RetrieveNode') if isinstance(inputs, dict) else None
        chain = build_retrieval_qa_chain(persist_dir, use_hf=use_hf)
        return simple_qa(chain, query)

    loader = Node('DocumentLoader', fn=loader_fn)
    index = Node('IndexNode', fn=index_fn)
    retrieve = Node('RetrieveNode', fn=retrieve_fn)
    llm = Node('LLMNode', fn=llm_fn)

    g.add_nodes([loader, index, retrieve, llm])
    g.add_edge('DocumentLoader','IndexNode')
    g.add_edge('IndexNode','RetrieveNode')
    g.add_edge('RetrieveNode','LLMNode')

    outputs = g.run(inputs={'DocumentLoader': uploaded_files})
    return outputs.get('LLMNode', 'No output produced')
