"""Construct the background-independent MFA Sankey accounting tables."""

import json
from hashlib import sha256

import pandas as pd

_NODE_COORDINATES = (
    "foreground_scenario",
    "year",
    "region",
    "material",
    "good",
    "element",
)


def _row_identity(values: tuple[object, ...]) -> str:
    """Build a stable identifier from the coordinates of a Sankey node or edge."""
    return sha256(
        json.dumps(
            values,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_mfa_sankey(
    model_flows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return nonzero final polymer edges and their nodes.

    The graph retains nonzero flow rows;
    other amounts and their explicit nominal element basis remain unchanged.
    Fabrication and end-of-life nodes pool goods within each remaining coordinate.
    Edges retain their original good; use and final-goods markets remain separate
    by good, as do all other processes with an explicit good coordinate.
    """

    reporting = model_flows.loc[model_flows["year"].between(2019, 2100)]
    selected = reporting.loc[reporting["amount"].ne(0.0)]
    source = selected.loc[:, _NODE_COORDINATES].copy()
    source["process"] = selected["source_process"].to_numpy()
    target = selected.loc[:, _NODE_COORDINATES].copy()
    target["process"] = selected["target_process"].to_numpy()
    # These processes connect aggregate flows and flows distinguished by end-use group.
    source.loc[source["process"].isin(("fabrication", "eol")), "good"] = "all"
    target.loc[target["process"].isin(("fabrication", "eol")), "good"] = "all"
    node_keys = [*_NODE_COORDINATES, "process"]
    nodes = pd.concat([source, target], ignore_index=True).drop_duplicates(node_keys)
    nodes["node_id"] = [
        _row_identity(tuple(getattr(row, column) for column in node_keys))
        for row in nodes.itertuples(index=False)
    ]
    edges = selected.reset_index(drop=True)
    edges["source_node_id"] = source.merge(nodes, on=node_keys, how="left")["node_id"].to_numpy()
    edges["target_node_id"] = target.merge(nodes, on=node_keys, how="left")["node_id"].to_numpy()
    edges["edge_id"] = [
        _row_identity(
            (
                row.source_node_id,
                row.target_node_id,
                row.source_flow_name,
            )
        )
        for row in edges.itertuples(index=False)
    ]

    return edges, nodes
