"""Early technology supplier selection preserves heat quantities through LCI."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)


@pytest.mark.parametrize("year", [2020, 2025, 2030])
def test_dac_materials_keep_their_distinct_source_heat_suppliers(tmp_path: Path, year: int) -> None:
    """Preserve PEI and silica quantities and their distinct heat suppliers."""
    catalogue = load_provider_catalogue()
    service = next(
        row
        for row in cast(list[dict[str, Any]], catalogue["services"])
        if row["provider_service_id"] == "methane_atmospheric_dac"
    )
    provider = PreparedProviderAuthority(
        identity="bounded_dac_materials",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": [{**service, "frontier_service": True}], "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="dac_materials",
        biosphere_database="biosphere",
        source_database="ordinary",
    )
    with selected_brightway_project(names) as bd:
        bd.geomapping.add(["World"])
        bd.Database("biosphere").write(
            {
                ("biosphere", "co2"): {
                    "name": "Carbon dioxide, fossil",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
            },
            searchable=False,
        )
        ordinary = {
            ("ordinary", code): {
                **_process(
                    "ordinary",
                    code,
                    "market for heat, central or small-scale, natural gas",
                    "heat, central or small-scale, natural gas",
                    location=location,
                    exchanges=[
                        {"input": ("biosphere", "co2"), "type": "biosphere", "amount": emission}
                    ],
                ),
                "unit": "megajoule",
            }
            for code, location, emission in (
                ("european_heat", "Europe without Switzerland", 0.1),
                ("row_heat", "RoW", 0.2),
            )
        }
        ordinary["ordinary", "methane_twin"] = _process(
            "ordinary",
            "methane_twin",
            "methane, from electrochemical methanation, with carbon from atmosphere",
            "methane, from electrochemical methanation",
            location="EUR",
        )
        bd.Database("ordinary").write(ordinary, searchable=False)
        technology: dict[tuple[str, str], dict[str, Any]] = {
            ("technology", code): _process(
                "technology",
                code,
                name,
                product,
                location="EUR",
                production_amount=production,
                exchanges=[
                    {"input": ("technology", child), "type": "technosphere", "amount": amount}
                    for child, amount in inputs
                ],
            )
            for code, name, product, production, inputs in (
                (
                    "methane",
                    "methane, from electrochemical methanation, with carbon from atmosphere",
                    "methane, from electrochemical methanation",
                    1.0,
                    [("capture", 2.0)],
                ),
                (
                    "capture",
                    "carbon dioxide, captured, with a sorbent-based direct air capture system, "
                    "100ktCO2",
                    "carbon dioxide, captured",
                    1.0,
                    [("sorbent", 0.5)],
                ),
                (
                    "sorbent",
                    "amine-based silica production, for sorbent-based direct air capture system",
                    "amine-based silica",
                    1.0,
                    [("pei", 0.4), ("silica", 0.6)],
                ),
                (
                    "pei",
                    "polyethyleneimine (PEI) production, "
                    "for sorbent-based direct air capture system",
                    "polyethyleneimine",
                    2.0,
                    [("shared_heat", 7.0)],
                ),
                (
                    "silica",
                    "silica gel production, for sorbent-based direct air capture system",
                    "silica gel",
                    1.0,
                    [("shared_heat", 5.0)],
                ),
            )
        }
        technology["technology", "shared_heat"] = {
            **_process(
                "technology",
                "shared_heat",
                "market for heat, for buildings",
                "heat, central or small-scale",
                location="EUR",
                exchanges=[{"input": ("biosphere", "co2"), "type": "biosphere", "amount": 2.0}],
            ),
            "unit": "megajoule",
        }
        technology["technology", "pei"]["exchanges"][1].update(
            {
                "uncertainty type": 3,
                "loc": 7.0,
                "scale": 0.4,
                "comment": "invented source heat distribution",
            }
        )
        technology["technology", "pei"]["exchanges"].extend(
            [
                {
                    "input": ("biosphere", "co2"),
                    "type": "biosphere",
                    "amount": 0.25,
                    "unit": "kilogram",
                    "uncertainty type": 3,
                    "loc": 0.25,
                    "scale": 0.025,
                    "comment": "invented direct PEI emission distribution",
                },
                {
                    "input": ("ordinary", "european_heat"),
                    "type": "technosphere",
                    "amount": 0.5,
                    "unit": "megajoule",
                    "uncertainty type": 3,
                    "loc": 0.5,
                    "scale": 0.1,
                    "comment": "invented PEI heat already linked to the ordinary supply",
                },
            ]
        )
        bd.Database("technology").write(technology, searchable=False)
        original = {
            node.key: [dict(e) for e in node.exchanges()] for node in bd.Database("technology")
        }
        source = "ordinary" if year < 2030 else "technology"
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=source,
            technology_database="technology",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache(source, "biosphere"),
        )
        root = bd.get_node(
            database=frontier.database_name, code=provider_service_code("methane_atmospheric_dac")
        )
        calculation = LCA({root: 1.0})
        calculation.lci()
        # One methane unit uses 2 capture * 0.5 sorbent * 0.4 PEI / 2 kg PEI
        # output. The added direct emission and ordinary heat retain their kg
        # CO2 and MJ quantities, including 0.1 kg CO2 per MJ of ordinary heat.
        added_emissions = (2 * 0.5 * 0.4 / 2) * (0.25 + 0.5 * 0.1)
        assert float(calculation.inventory.sum()) == pytest.approx(
            (0.74 if year < 2030 else 8.8) + added_emissions
        )
        assert {
            node.key: [dict(e) for e in node.exchanges()] for node in bd.Database("technology")
        } == original
        copies = {
            node["provider_source_code"]: node
            for node in bd.Database(frontier.database_name)
            if node.get("hybrid_provider_technology_clone")
        }
        if year < 2030:
            assert set(copies) == {"methane", "capture", "sorbent", "pei", "silica"}
            for code, supplier in (("pei", "european_heat"), ("silica", "row_heat")):
                expected = [
                    {k: v for k, v in row.items() if k != "output"}
                    for row in original["technology", code]
                ]
                expected[0]["input"] = copies[code].key
                expected[1]["input"] = ("ordinary", supplier)
                assert [
                    {k: v for k, v in dict(row).items() if k != "output"}
                    for row in copies[code].exchanges()
                ] == expected
            assert root["provider_source_role"] == "premise_only_provider"
            bindings = [
                row
                for row in frontier.technology_rows
                if row["ledger_entry_type"] == "early_provider_ordinary_upstream_rebinding"
            ]
            assert {
                (row["provider_parent_code"], row["ordinary_provider_code"]) for row in bindings
            } == {
                ("pei", "european_heat"),
                ("silica", "row_heat"),
            }
        else:
            assert not copies
            assert root["provider_source_role"] == "matching_member_premise_provider"
            assert root["general_background_anchor_year"] == year
