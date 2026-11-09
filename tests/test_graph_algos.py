"""Unit tests for get_path_in_graph.

Contract assumed by these tests:
  * get_path_in_graph(initial_node, target_node) returns a tuple
    (found, path).
  * found is True if target_node is reachable from initial_node by following
    Node.connections (edges are directed), otherwise False.
  * path is the list of Node objects from initial_node to target_node, both
    inclusive, or an empty list when there is no path.
  * When several paths exist, any valid simple path is accepted, so the tests
    do not depend on DFS vs BFS ordering.
"""

import pytest

from nptel_ai_course.graph_algos import get_path_in_graph
from nptel_ai_course.graph_models import Node


def build_graph(edges, extra_nodes=()):
    """Build a graph from (src, dst) name pairs and return {name: Node}.

    All nodes are created first and wired afterwards, so cycles and
    self-loops are easy to express. extra_nodes adds nodes with no edges.
    """
    names = list(extra_nodes)
    for src, dst in edges:
        names.extend([src, dst])
    nodes = {name: Node(name, []) for name in dict.fromkeys(names)}
    for src, dst in edges:
        nodes[src].connections.append(nodes[dst])
    return nodes


def assert_valid_path(path, start, end):
    """Assert path is a simple path start -> end along existing edges."""
    assert isinstance(path, list)
    assert path, "path must not be empty"
    assert path[0] is start, "path must begin at the initial node"
    assert path[-1] is end, "path must end at the target node"
    assert len({id(n) for n in path}) == len(path), "path repeats a node"
    for cur, nxt in zip(path, path[1:], strict=False):
        assert any(nxt is c for c in cur.connections), (
            f"no edge {cur.val} -> {nxt.val}"
        )


# --------------------------------------------------------------------------
# Return shape
# --------------------------------------------------------------------------


def test_returns_tuple_of_bool_and_list_when_path_exists():
    g = build_graph([("A", "B")])

    result = get_path_in_graph(g["A"], g["B"])

    assert isinstance(result, tuple)
    assert len(result) == 2
    found, path = result
    assert found is True
    assert isinstance(path, list)


def test_returns_false_and_empty_list_when_no_path():
    g = build_graph([("A", "B")], extra_nodes=["C"])

    found, path = get_path_in_graph(g["A"], g["C"])

    assert found is False
    assert path == []


# --------------------------------------------------------------------------
# Paths that exist and are unique (exact path can be asserted)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edges", "start", "end", "expected"),
    [
        pytest.param([("A", "B")], "A", "B", ["A", "B"], id="direct-neighbour"),
        pytest.param(
            [("A", "B"), ("B", "C"), ("C", "D")],
            "A",
            "D",
            ["A", "B", "C", "D"],
            id="linear-chain",
        ),
        pytest.param(
            [("A", "B"), ("B", "C"), ("C", "D")],
            "B",
            "D",
            ["B", "C", "D"],
            id="starts-mid-chain",
        ),
        pytest.param(
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")],
            "A",
            "E",
            ["A", "C", "E"],
            id="dead-end-branch-explored-first",
        ),
        pytest.param(
            [("A", "B"), ("A", "C"), ("C", "D")],
            "A",
            "C",
            ["A", "C"],
            id="target-is-later-sibling",
        ),
        pytest.param(
            [("A", "B"), ("B", "C"), ("B", "D")],
            "A",
            "D",
            ["A", "B", "D"],
            id="target-is-later-child",
        ),
        pytest.param(
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "E")],
            "A",
            "E",
            ["A", "C", "E"],
            id="shared-dead-end-node-not-revisited",
        ),
        pytest.param(
            [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")],
            "A",
            "C",
            ["A", "B", "C"],
            id="bidirectional-edges",
        ),
        pytest.param(
            [("A", "A"), ("A", "B")],
            "A",
            "B",
            ["A", "B"],
            id="self-loop-on-initial-node",
        ),
        pytest.param(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")],
            "A",
            "D",
            ["A", "B", "C", "D"],
            id="cycle-before-target",
        ),
    ],
)
def test_finds_the_unique_path(edges, start, end, expected):
    g = build_graph(edges)

    found, path = get_path_in_graph(g[start], g[end])

    assert found is True
    assert path == [g[name] for name in expected]


def test_finds_path_in_long_chain():
    names = [f"n{i}" for i in range(50)]
    g = build_graph(list(zip(names, names[1:], strict=False)))

    found, path = get_path_in_graph(g[names[0]], g[names[-1]])

    assert found is True
    assert path == [g[name] for name in names]


# --------------------------------------------------------------------------
# Several valid paths exist: only check the result is a valid path
# --------------------------------------------------------------------------


def test_diamond_graph_returns_a_valid_path():
    g = build_graph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])

    found, path = get_path_in_graph(g["A"], g["D"])

    assert found is True
    assert_valid_path(path, g["A"], g["D"])
    assert len(path) == 3


def test_graph_with_shortcut_returns_a_valid_path():
    g = build_graph(
        [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"), ("B", "D")]
    )

    found, path = get_path_in_graph(g["A"], g["D"])

    assert found is True
    assert_valid_path(path, g["A"], g["D"])


def test_dense_cyclic_graph_returns_a_valid_path():
    names = ["A", "B", "C", "D", "E"]
    edges = [(a, b) for a in names for b in names if a != b]
    g = build_graph(edges)

    found, path = get_path_in_graph(g["A"], g["E"])

    assert found is True
    assert_valid_path(path, g["A"], g["E"])


# --------------------------------------------------------------------------
# Initial node is the target
# --------------------------------------------------------------------------


def test_initial_node_equals_target_node():
    g = build_graph([("A", "B")])

    found, path = get_path_in_graph(g["A"], g["A"])

    assert found is True
    assert path == [g["A"]]


def test_initial_node_equals_target_node_inside_a_cycle():
    g = build_graph([("A", "B"), ("B", "A")])

    found, path = get_path_in_graph(g["A"], g["A"])

    assert found is True
    assert path == [g["A"]]


def test_isolated_node_is_its_own_path():
    g = build_graph([], extra_nodes=["A"])

    found, path = get_path_in_graph(g["A"], g["A"])

    assert found is True
    assert path == [g["A"]]


# --------------------------------------------------------------------------
# No path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edges", "extra_nodes", "start", "end"),
    [
        pytest.param([], ["A", "B"], "A", "B", id="two-isolated-nodes"),
        pytest.param(
            [("A", "B")], [], "B", "A", id="edge-points-the-wrong-way"
        ),
        pytest.param(
            [("A", "B"), ("C", "D")],
            [],
            "A",
            "D",
            id="different-components",
        ),
        pytest.param(
            [("A", "B"), ("B", "C")],
            ["Z"],
            "A",
            "Z",
            id="target-not-connected-to-anything",
        ),
        pytest.param(
            [("A", "B"), ("B", "C")],
            ["Z"],
            "Z",
            "A",
            id="initial-has-no-connections",
        ),
        pytest.param(
            [("A", "B"), ("B", "C"), ("C", "A")],
            ["D"],
            "A",
            "D",
            id="cycle-and-unreachable-target",
        ),
        pytest.param(
            [("A", "A")], ["B"], "A", "B", id="self-loop-and-unreachable"
        ),
        pytest.param(
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
            ["E"],
            "A",
            "E",
            id="shared-nodes-and-unreachable-target",
        ),
    ],
)
def test_reports_no_path(edges, extra_nodes, start, end):
    g = build_graph(edges, extra_nodes=extra_nodes)

    found, path = get_path_in_graph(g[start], g[end])

    assert found is False
    assert path == []


# --------------------------------------------------------------------------
# Identity, purity and repeatability
# --------------------------------------------------------------------------


def test_path_contains_the_node_objects_themselves():
    g = build_graph([("A", "B"), ("B", "C")])

    _, path = get_path_in_graph(g["A"], g["C"])

    assert [id(n) for n in path] == [id(g[n]) for n in "ABC"]


def test_nodes_are_matched_by_identity_not_by_value():
    start = Node("start", [])
    twin_1 = Node("same", [])
    twin_2 = Node("same", [])
    start.connections.extend([twin_1, twin_2])

    found, path = get_path_in_graph(start, twin_2)

    assert found is True
    assert len(path) == 2
    assert path[0] is start
    assert path[1] is twin_2


def test_does_not_modify_the_graph():
    g = build_graph(
        [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "A")]
    )
    before = {name: list(node.connections) for name, node in g.items()}

    get_path_in_graph(g["A"], g["D"])
    get_path_in_graph(g["A"], g["A"])

    for name, node in g.items():
        assert len(node.connections) == len(before[name])
        for kept, current in zip(before[name], node.connections, strict=True):
            assert kept is current


def test_repeated_calls_give_the_same_answer():
    g = build_graph([("A", "B"), ("B", "C"), ("A", "D")], extra_nodes=["Z"])

    first = get_path_in_graph(g["A"], g["C"])
    second = get_path_in_graph(g["A"], g["C"])
    miss_first = get_path_in_graph(g["A"], g["Z"])
    miss_second = get_path_in_graph(g["A"], g["Z"])

    assert first == second
    assert miss_first == miss_second
