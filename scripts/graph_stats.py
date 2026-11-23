"""Graph and community statistics from GraphRAG index outputs.

Each subcommand works on GraphRAG output folders (the directory that contains
``relationships.parquet``, ``communities.parquet``, ``community_reports.parquet``).

Graph statistics
    python scripts/graph_stats.py graph --output <run>/output

    |V| and |E| are measured on the largest connected component (LCC) as returned
    by GraphRAG's ``stable_largest_connected_component`` -- self-loops are KEPT in
    |E|. K_m is the max core number after removing self-loops (as the k-core
    clustering does). Average degree and the share of degree-1 nodes use the
    same LCC.

Communities and coverage
    python scripts/graph_stats.py communities --output <kcore-run>/output \
        --reference <leiden-run>/output

    "# Tokens" is the total number of cl100k tokens over ALL community reports of
    the Leiden (reference) index. For a hierarchy level L, the communities used to
    answer a query are all communities at level L plus every leaf community
    (no children) at levels < L; "# Comm." counts them and "Coverage" is the share
    of reference tokens in their reports. Levels LF and L1 are printed.

Cluster-size statistics
    python scripts/graph_stats.py sizes --output <run>/output --heuristic RkH

    Re-runs the k-core hierarchy on the run's graph and reports the share of
    size-2 and size-3/4 leaf clusters.

Run everything from the data shipped in this repository
    python scripts/graph_stats.py repo

    data/graphs/<model>/<dataset>/relationships.parquet
    data/communities/<method>/<dataset>/community_reports.parquet
        <method> in {leiden, RkH, M2hC, MRC}; GPT-3.5-turbo runs.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

HEURISTICS = ("RkH", "M2hC", "MRC")


# --------------------------------------------------------------------------- graph
def _lcc(graph: nx.Graph) -> nx.Graph:
    """Largest connected component, identical to GraphRAG's stable LCC."""
    try:
        from graphrag.index.utils.stable_lcc import stable_largest_connected_component

        return stable_largest_connected_component(graph)
    except ImportError:  # graspologic not installed: equivalent networkx version
        g = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        g = nx.relabel_nodes(
            g, {n: html.unescape(n.upper().strip()) if isinstance(n, str) else n for n in g}
        )
        fixed = nx.Graph()
        fixed.add_nodes_from(sorted(g.nodes(data=True), key=lambda x: x[0]))
        edges = [(min(a, b), max(a, b), d) for a, b, d in g.edges(data=True)]
        fixed.add_edges_from(sorted(edges, key=lambda x: f"{x[0]} -> {x[1]}"))
        return fixed


def load_graph(output: Path) -> nx.Graph:
    rel = pd.read_parquet(output / "relationships.parquet")
    return nx.from_pandas_edgelist(rel, edge_attr=["weight"])


def graph_stats(output: Path) -> dict:
    lcc = _lcc(load_graph(output))
    n, m = lcc.number_of_nodes(), lcc.number_of_edges()  # self-loops kept
    deg1 = sum(1 for _, d in lcc.degree() if d == 1)
    no_loops = lcc.copy()
    no_loops.remove_edges_from(list(nx.selfloop_edges(no_loops)))
    return {
        "V": n,
        "E": m,
        "K_m": max(nx.core_number(no_loops).values()),
        "avg_degree": 2 * m / n,
        "deg1_pct": 100 * deg1 / n,
    }


# ------------------------------------------------------------------- communities
def _report_tokens(output: Path) -> pd.DataFrame:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    rep = pd.read_parquet(output / "community_reports.parquet")
    rep["tokens"] = [len(enc.encode(str(t))) for t in rep["full_content"]]
    return rep


def community_stats(output: Path, reference: Path) -> dict[int, tuple[int, float]]:
    """Per level L: (#communities used at L, coverage % of reference report tokens)."""
    total = _report_tokens(reference)["tokens"].sum()
    rep = _report_tokens(output)
    result, leaf_count, leaf_tokens = {}, 0, 0
    for level, group in rep.groupby("level"):
        result[int(level)] = (
            leaf_count + len(group),
            100 * (leaf_tokens + group["tokens"].sum()) / total,
        )
        leaves = group[group["children"].apply(len) == 0]
        leaf_count += len(leaves)
        leaf_tokens += leaves["tokens"].sum()
    return result


# ------------------------------------------------------------------------- sizes
def size_stats(output: Path, heuristic: str) -> dict:
    import graphrag.index.operations.kcore_cluster_graph as kcg

    kcg.stable_largest_connected_component = _lcc
    with contextlib.redirect_stdout(io.StringIO()):
        clusters = kcg.kcore_cluster_graph(load_graph(output), True, heuristic)
    parents = {c[2] for c in clusters}
    leaf_sizes = Counter(len(c[3]) for c in clusters if c[1] not in parents)
    n_leaf = sum(leaf_sizes.values())
    return {
        "clusters": len(clusters),
        "leaf_clusters": n_leaf,
        "leaf_size2_pct": 100 * leaf_sizes[2] / n_leaf,
        "leaf_size3_4_pct": 100 * (leaf_sizes[3] + leaf_sizes[4]) / n_leaf,
    }


# --------------------------------------------------------------------------- cli
def _print_graph(name: str, s: dict) -> None:
    print(
        f"{name:32} |V|={s['V']:6}  |E|={s['E']:6}  K_m={s['K_m']:3}"
        f"  avg_deg={s['avg_degree']:.2f}  deg1={s['deg1_pct']:.1f}%"
    )


def _print_comm(name: str, levels: dict[int, tuple[int, float]]) -> None:
    lf = max(levels)
    for tag, lvl in (("LF", lf), ("L1", lf - 1)):
        if lvl in levels:
            c, cov = levels[lvl]
            print(f"{name:32} {tag} (level {lvl:2}): #Comm={c:5}  Coverage={cov:6.2f}%")


def run_repo(data: Path) -> None:
    t1, t5 = data / "graphs", data / "communities"
    for model in sorted(p.name for p in t1.iterdir() if p.is_dir()):
        print(f"\n===== Graph statistics: {model}")
        for ds in sorted((t1 / model).iterdir()):
            _print_graph(ds.name, graph_stats(ds))

    print("\n===== Communities and coverage (GPT-3.5-turbo)")
    for ds in ("podcast", "news", "semiconductor"):
        ref = t5 / "leiden" / ds
        total = _report_tokens(ref)["tokens"].sum()
        print(f"-- {ds}: # Tokens = {total:,}")
        for h in HEURISTICS:
            _print_comm(h, community_stats(t5 / h / ds, ref))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("graph")
    g.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("communities")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--reference", type=Path, required=True, help="Leiden output folder")
    s = sub.add_parser("sizes")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--heuristic", choices=HEURISTICS, default="RkH")
    r = sub.add_parser("repo")
    r.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = p.parse_args()

    if args.cmd == "graph":
        _print_graph(str(args.output), graph_stats(args.output))
    elif args.cmd == "communities":
        _print_comm(str(args.output), community_stats(args.output, args.reference))
    elif args.cmd == "sizes":
        print(size_stats(args.output, args.heuristic))
    else:
        run_repo(args.data)


if __name__ == "__main__":
    main()
