"""Ranking visualisation helpers shared by both notebooks.

These are presentation-only — the analyses live in :mod:`mcda_common.promethee`
and :mod:`mcda_common.electre`. The graph layout uses Graphviz via
:func:`networkx.drawing.nx_agraph.graphviz_layout`, which keeps hierarchical
preference graphs readable without manual coordinate fiddling.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout


def find_nodes_groups(ranking: pd.DataFrame) -> list[list[str]]:
    """Cliques of mutually-indifferent alternatives in an outranking matrix."""
    nodes = ranking.index.tolist()
    indifference = ranking & ranking.T
    edges = [
        (nodes[i], nodes[j])
        for i, j in np.stack(np.nonzero(indifference)).T.tolist()
        if i != j
    ]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return list(nx.clique.find_cliques(g))


def display_ranking(ranking: pd.DataFrame, title: str) -> None:
    """Render the outranking matrix as a layered preference graph."""
    nodes_groups = find_nodes_groups(ranking)
    nodes = ranking.index.tolist()
    edges = [
        (nodes[i], nodes[j])
        for i, j in np.stack(np.nonzero(ranking)).T.tolist()
        if i != j
    ]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # Collapse each indifference clique into a single node labelled with all
    # its members, then take the transitive reduction so the layered layout
    # only shows immediate preferences.
    names_mapping: dict[str, str] = {}
    for node_group in nodes_groups:
        first, *others = node_group
        for member in others:
            g.remove_node(member)
        names_mapping[first] = "\n".join(node_group)

    g = nx.relabel_nodes(g, names_mapping)
    g = nx.transitive_reduction(g)

    layout = graphviz_layout(g, prog="dot")
    plt.title(title)
    nx.draw(
        g,
        layout,
        with_labels=True,
        arrows=False,
        node_shape="s",
        node_color="none",
        bbox={"facecolor": "white", "edgecolor": "black"},
    )
    plt.show()


def plot_single_marginal_preference_function(
    function: Callable[[float, float, float], float],
    p: float,
    q: float,
    ax: Sequence[plt.Axes],
    x_min: float = -2.0,
    x_max: float = 3.0,
) -> None:
    """Side-by-side plot of the computed marginal preference function and its expected shape."""
    vectorised = np.vectorize(
        partial(function, indifference_threshold=q, preference_threshold=p),
        otypes=[float],
    )
    x = np.linspace(x_min, x_max, 300)

    ax[0].plot(x, vectorised(x))
    ax[1].plot([x_min, q, p, x_max], [0, 0, 1, 1])

    ticks: list[float] = [0]
    labels: list[str] = ["0"]
    if q == 0:
        labels[0] = "q=0"
    else:
        ticks.append(q)
        labels.append("q")
    if p == q:
        labels[-1] = "q=p" + labels[-1][1:]
    else:
        ticks.append(p)
        labels.append("p")

    ax[0].set_xticks(ticks, labels)
    ax[1].set_xticks(ticks, labels)


def plot_marginal_preference_function(
    function: Callable[[float, float, float], float],
) -> None:
    """4-row test grid for the marginal preference function under different (q, p) regimes."""
    fig, ax = plt.subplots(4, 2)
    ax[0, 0].set_title("Computed")
    ax[0, 1].set_title("Expected")

    plot_single_marginal_preference_function(function, q=0, p=2, ax=ax[0])
    plot_single_marginal_preference_function(function, q=1, p=2, ax=ax[1])
    plot_single_marginal_preference_function(function, q=1, p=1, ax=ax[2])
    plot_single_marginal_preference_function(function, q=0, p=0, ax=ax[3])

    fig.tight_layout()


# Backward-compatible names used by the original utils.py / notebooks.
test_single_marginal_preference_function = plot_single_marginal_preference_function
test_marginal_preference_function = plot_marginal_preference_function
