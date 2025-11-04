from graphviz import Digraph

from micrograd.engine import Value


def build_graph(root: Value) -> tuple[set[Value], set[tuple[Value, ...]]]:
    nodes, edges = set(), set()

    def build(node) -> None:
        if node not in nodes:
            nodes.add(node)
            for child in node._children:
                edges.add((child, node))
                build(child)

    build(root)
    return (nodes, edges)


def draw_graph(root: Value) -> Digraph:
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = build_graph(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid, label="{ %s | data %.4f | grad %f }" % (n.label, n.data, n.grad), shape="record"
        )

        if op := n._op:
            dot.node(name=uid+op, label=op)
            dot.edge(uid+op, uid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2))+n2._op)

    return dot

