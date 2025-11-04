
"""Lightweight in-project LangGraph-compatible implementation.
This provides Graph and Node classes with .add_node, .add_edge, and .run(inputs)
to simulate a LangGraph workflow for portability and demo purposes.
"""
from collections import defaultdict

class Node:
    def __init__(self, name, fn=None):
        self.name = name
        self.fn = fn if fn is not None else (lambda *args, **kwargs: None)
        self.inputs = []
        self.outputs = None

    def run(self, inputs):
        try:
            self.outputs = self.fn(inputs)
        except TypeError:
            # allow functions that take multiple args in legacy style
            self.outputs = self.fn(**inputs) if isinstance(inputs, dict) else self.fn(inputs)
        return self.outputs

class Graph:
    def __init__(self, name='graph'):
        self.name = name
        self._nodes = {}
        self._edges = defaultdict(list)  # from -> [to]
        self._reverse = defaultdict(list)  # to -> [from]

    def add_node(self, node):
        self._nodes[node.name] = node

    def add_nodes(self, nodes):
        for n in nodes:
            self.add_node(n)

    def add_edge(self, src, dst):
        if src not in self._nodes or dst not in self._nodes:
            raise ValueError('Both nodes must be added before linking')
        self._edges[src].append(dst)
        self._reverse[dst].append(src)

    def run(self, inputs=None):
        # naive topological-ish runner: run nodes whose inputs are available.
        inputs = inputs or {}
        results = {}
        pending = set(self._nodes.keys())
        # seed nodes with provided inputs if keys match node names
        for k,v in inputs.items():
            if k in self._nodes:
                self._nodes[k].outputs = v
                results[k] = v
                pending.discard(k)
        # Keep running until no pending nodes can run
        progressed = True
        while pending and progressed:
            progressed = False
            for name in list(pending):
                # check if all predecessors have outputs (or no predecessors)
                preds = self._reverse.get(name, [])
                ready = all((p in results) for p in preds)
                if ready:
                    # collect inputs from predecessors
                    inp = {}
                    for p in preds:
                        inp[p] = results[p]
                    # also pass global inputs matching node name
                    if name in inputs:
                        inp[name] = inputs[name]
                    node = self._nodes[name]
                    out = node.run(inp)
                    results[name] = out
                    pending.discard(name)
                    progressed = True
        return results
