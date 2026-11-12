"""Binary graph reduction through excluded sensor vertices, never test data."""

import csv
from collections import deque

import numpy as np


def reduce_edges(edges, retained, known):
    """Stop at retained vertices and record witness paths."""
    retained = set(retained)
    known = set(known)
    nodes = {n for e in edges for n in e}
    if nodes - known:
        raise ValueError(f"Unknown graph IDs: {sorted(nodes - known)}")
    adj = {n: set() for n in known}
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    paths = {}
    for start in sorted(retained):
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            for other in sorted(adj[node]):
                if other in seen:
                    continue
                seen.add(other)
                route = path + [other]
                if other in retained:
                    key = tuple(sorted((start, other)))
                    canonical = route if start == key[0] else route[::-1]
                    if key not in paths or len(canonical) < len(paths[key]):
                        paths[key] = canonical
                else:
                    queue.append((other, route))
    return paths


def reduce_graph(path, names, known):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    if any(
        len(r) != 2 or any(x != x.strip() or not x for x in r) for r in rows
    ):
        raise ValueError("Malformed edge IDs")
    edges = {tuple(sorted(r)) for r in rows if r[0] != r[1]}
    paths = reduce_edges(edges, names, known)
    idx = {n: i for i, n in enumerate(names)}
    a = np.zeros((len(names), len(names)))
    for u, v in paths:
        a[idx[u], idx[v]] = a[idx[v], idx[u]] = 1
    direct = {e for e in edges if set(e) <= set(names)}
    audit = {
        "policy": (
            "binary undirected; internal path vertices must be "
            "excluded sensors"
        ),
        "selection_source": (
            "train only; no traffic values used for " "graph traversal"
        ),
        "raw_rows": len(rows),
        "unique_nonself_edges": len(edges),
        "direct_edges_retained": len(direct),
        "final_edges": len(paths),
        "isolated_sensors": [n for n, i in idx.items() if not a[i].any()],
        "added_edges": [
            {
                "endpoints": list(e),
                "witness_path": paths[e],
                "original_hops": len(paths[e]) - 1,
            }
            for e in sorted(set(paths) - direct)
        ],
        "final_edge_list": [list(e) for e in sorted(paths)],
    }
    return a, audit


def self_test():
    chain = {("A", "B"), ("B", "C"), ("C", "D")}
    assert set(reduce_edges(chain, ["A", "C", "D"], "ABCD")) == {
        ("A", "C"),
        ("C", "D"),
    }
    assert set(reduce_edges(chain, ["A", "D"], "ABCD")) == {("A", "D")}
    assert set(reduce_edges(chain, "ABCD", "ABCD")) == chain
    branch = {("A", "X"), ("B", "X"), ("C", "X")}
    assert set(reduce_edges(branch, "ABC", "ABCX")) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    }
    assert not reduce_edges({("A", "B")}, ["A", "C"], "ABC")
    try:
        reduce_edges({("A", "unknown")}, ["A"], ["A"])
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown intermediate vertex accepted")
