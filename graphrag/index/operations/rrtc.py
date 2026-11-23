# Copyright (c) 2026 Jakir Hossain, Ahmet Erdem Sariyuce.
# Licensed under the MIT License

"""Round-Robin Token-Constrained Selection (RRTC).

Selects a subset of edges from the communities used at a hierarchy level:
edges are ranked within each community by the combined degree of their
endpoints, and picked round-robin across communities, from higher to lower
k-shells, until the edge budget is reached.
"""

import pandas as pd


def rrtc_select(
    communities: pd.DataFrame,
    relationships: pd.DataFrame,
    budget: float,
    level: int | None = None,
) -> pd.DataFrame:
    """Return the relationships selected by RRTC.

    budget: fraction of edges to keep, e.g. 0.8.
    level: hierarchy level (default: leaf level LF); uses the communities at
        this level plus the leaf communities of lower levels.
    """
    if level is None:
        level = communities["level"].max()
    leaf = communities["children"].apply(len) == 0
    used = communities[
        (communities["level"] == level) | ((communities["level"] < level) & leaf)
    ].sort_values(["level", "community"], ascending=[False, True])

    rel = relationships.set_index("id")
    queues, seen = [], set()
    for ids in used["relationship_ids"]:
        ids = [i for i in ids if i in rel.index and i not in seen]
        seen.update(ids)
        ranked = rel.loc[ids].sort_values(
            ["combined_degree", "human_readable_id"], ascending=[False, True]
        )
        queues.append(list(ranked.index))

    limit = int(budget * len(seen))
    selected, depth = [], 0
    while len(selected) < limit:
        for queue in queues:
            if depth < len(queue) and len(selected) < limit:
                selected.append(queue[depth])
        depth += 1
    return rel.loc[selected].reset_index()
