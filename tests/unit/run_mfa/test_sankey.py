"""Sankey nodes and edges preserve the actual MFA account."""

import json
from importlib import resources

import pandas as pd
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from material_recipe_fixtures import material_source_members

from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.run_mfa.model.flows import NOMINAL_ELEMENT_BASIS
from pytoce.run_mfa.model.sankey import build_mfa_sankey


def test_sankey_preserves_physical_flow_coordinates_and_amounts() -> None:
    """Retain every reported nonzero flow and its exact amount between unique nodes."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    state, inputs = empty_model("HDPE")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[1, region, good] = 1e-9
    state.stock_by_cohort[1:, 1, region, good] = 1e-9
    model = evaluate_mfa(state, inputs, recipe).model_flows

    edges, nodes = build_mfa_sankey(model)

    expected = model.loc[model.year.between(2019, 2100) & model.amount.ne(0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(edges.loc[:, model.columns], expected, check_exact=True)
    assert set(edges.source_node_id) | set(edges.target_node_id) == set(nodes.node_id)
    assert nodes.node_id.is_unique
    assert edges.edge_id.is_unique


def test_sankey_connects_pooled_processes_without_losing_flow_coordinates() -> None:
    """Connect fabrication and waste pooling while retaining use and trade by good."""
    movements = (
        ("processing", "fabrication", "all", 3.0),
        ("fabrication", "use", "Packaging", 1.0),
        ("fabrication", "use", "Transportation", 1.0),
        ("fabrication", "good_market", "Packaging", 1.0),
        ("good_market", "sysenv", "Packaging", 1.0),
        ("sysenv", "good_market", "Transportation", 2.0),
        ("good_market", "use", "Transportation", 2.0),
        ("use", "eol", "Packaging", 1.0),
        ("use", "eol", "Transportation", 2.0),
        ("eol", "collected", "all", 2.0),
        ("eol", "mismanaged", "all", 1.0),
    )
    coordinates = (
        ("scenario_a", 2019, "EUR", "HDPE", "C"),
        ("scenario_b", 2019, "EUR", "HDPE", "C"),
        ("scenario_a", 2100, "EUR", "HDPE", "C"),
        ("scenario_a", 2019, "USA", "HDPE", "C"),
        ("scenario_a", 2019, "EUR", "PET", "C"),
        ("scenario_a", 2019, "EUR", "HDPE", "Other Elements"),
    )
    records = [
        {
            "foreground_scenario": scenario,
            "year": year,
            "region": region,
            "material": material,
            "element": element,
            "good": good,
            "source_process": source,
            "target_process": target,
            "source_flow_name": f"{source} => {target}",
            "amount": amount,
            "unit": "Mt",
            "basis": NOMINAL_ELEMENT_BASIS,
            "custody": "h12_physical_custody",
        }
        for scenario, year, region, material, element in coordinates
        for source, target, good, amount in movements
    ]
    records.extend(
        {**records[0], "year": year, "amount": amount}
        for year, amount in ((2018, 3.0), (2020, 0.0))
    )
    model = pd.DataFrame.from_records(records)
    original = model.copy(deep=True)

    edges, nodes = build_mfa_sankey(model)

    expected = model.loc[model.year.between(2019, 2100) & model.amount.ne(0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(model, original, check_exact=True)
    pd.testing.assert_frame_equal(edges.loc[:, model.columns], expected, check_exact=True)
    assert edges.edge_id.is_unique
    assert nodes.node_id.is_unique
    assert set(edges.source_node_id) | set(edges.target_node_id) == set(nodes.node_id)
    dimensions = ["foreground_scenario", "year", "region", "material", "element"]
    for coordinate, group in edges.groupby(dimensions, sort=False):
        selected_nodes = nodes.set_index(dimensions).loc[[coordinate]]
        for process in ("fabrication", "eol"):
            pooled = selected_nodes.loc[selected_nodes.process.eq(process)]
            assert pooled.good.tolist() == ["all"]
            node_id = pooled.node_id.iloc[0]
            incoming = group.loc[group.target_process.eq(process)]
            outgoing = group.loc[group.source_process.eq(process)]
            assert set(incoming.target_node_id) == {node_id}
            assert set(outgoing.source_node_id) == {node_id}
            assert incoming.amount.sum() == outgoing.amount.sum()
        for process in ("use", "good_market", "sysenv"):
            separated = selected_nodes.loc[selected_nodes.process.eq(process)]
            assert set(separated.good) == {"Packaging", "Transportation"}

    schema = json.loads(
        resources.files("pytoce.run_mfa.data.schemas")
        .joinpath("mfa_result_schema_v1.json")
        .read_text(encoding="utf-8")
    )
    artifacts = {entry["path"]: entry for entry in schema["artifacts"]}
    assert set(artifacts["sankey/edges.parquet"]["required_columns"]) <= set(edges.columns)
    assert set(artifacts["sankey/nodes.parquet"]["required_columns"]) <= set(nodes.columns)
