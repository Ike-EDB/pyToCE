"""Fixed wind and metal suppliers in the early PEM hydrogen inventory."""

from pathlib import Path

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
def test_pem_preserves_metal_mixtures_and_uses_declared_country_suppliers(
    tmp_path: Path, year: int
) -> None:
    """Solve an invented wind/PEM graph with distinct metal mixes and signed waste."""
    catalogue = load_provider_catalogue()
    xylene = next(
        row for row in catalogue["services"] if row["provider_service_id"] == "p_xylene_biogenic"
    )
    hydrogen = next(
        row
        for row in xylene["construction"]["components"]
        if row.get("component_id") == "green_hydrogen"
    )
    service = {
        **xylene,
        "provider_service_id": "bounded_wind_hydrogen",
        "name": "bounded wind hydrogen",
        "reference_product": "hydrogen, gaseous, 30 bar",
        "construction": {
            "kind": "native_activity",
            "source_role": hydrogen["source_role"],
            "selector": hydrogen["selector"],
        },
    }
    provider = PreparedProviderAuthority(
        identity="bounded_pem_authority",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": [service], "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="pem",
        biosphere_database="biosphere",
        source_database="ordinary",
    )
    with selected_brightway_project(names) as bd:
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
        for database in ("ordinary", "technology"):
            supplies = [
                (
                    "wind",
                    "electricity production, wind, >3MW turbine, onshore",
                    "electricity, high voltage",
                    "kilowatt hour",
                    "DK",
                    1.0,
                    0.08 if database == "ordinary" else 0.8,
                ),
                (
                    "tailings",
                    "market for sulfidic tailings, generic",
                    "sulfidic tailings, generic",
                    "kilogram",
                    "GLO" if database == "ordinary" else "CAZ",
                    -2.0,
                    0.2 if database == "ordinary" else 2.0,
                ),
            ]
            if database == "ordinary":
                supplies.extend(
                    (
                        country,
                        "market group for electricity, medium voltage"
                        if country == "CA"
                        else "market for electricity, medium voltage",
                        "electricity, medium voltage",
                        "kilowatt hour",
                        country,
                        1.0,
                        emission,
                    )
                    for country, emission in (("CA", 0.1), ("ZA", 0.2), ("ZW", 0.3))
                )
            else:
                supplies.extend(
                    (
                        location,
                        "market group for electricity, medium voltage",
                        "electricity, medium voltage",
                        "kilowatt hour",
                        location,
                        1.0,
                        emission,
                    )
                    for location, emission in (("World", 10.0), ("SSA", 20.0))
                )
            data = {
                (database, code): {
                    **_process(
                        database,
                        code,
                        name,
                        product,
                        location=location,
                        production_amount=production,
                        exchanges=[
                            {
                                "input": ("biosphere", "co2"),
                                "type": "biosphere",
                                "amount": emission,
                            }
                        ],
                    ),
                    "unit": unit,
                }
                for code, name, product, unit, location, production, emission in supplies
            }
            if database == "technology":
                data[database, "hydrogen"] = _process(
                    database,
                    "hydrogen",
                    hydrogen["selector"]["name"],
                    "hydrogen, gaseous, 30 bar",
                    exchanges=[
                        {"input": (database, code), "type": "technosphere", "amount": amount}
                        for code, amount in (("wind", 4.0), ("stack", 0.5), ("stack", 0.25))
                    ],
                )
                data[database, "stack"] = {
                    **_process(
                        database,
                        "stack",
                        "electrolyzer production, 1MWe, PEM, Stack",
                        "electrolyzer, 1MWe, PEM, Stack",
                        exchanges=[
                            {"input": (database, metal), "type": "technosphere", "amount": amount}
                            for metal, amount in (("iridium", 0.2), ("ruthenium", 0.1))
                        ],
                    ),
                    "unit": "unit",
                }
                for metal, shares in (("iridium", (0.2, 0.5, 0.3)), ("ruthenium", (0.4, 0.1, 0.5))):
                    data[database, metal] = _process(
                        database,
                        metal,
                        f"market for {metal}",
                        metal,
                        location="World",
                        exchanges=[
                            {
                                "input": (database, f"{metal}_{country}"),
                                "type": "technosphere",
                                "amount": share,
                            }
                            for country, share in zip(("CA", "ZA", "ZW"), shares, strict=True)
                        ],
                    )
                    for country, energy in (("CA", 2.0), ("ZA", 3.0), ("ZW", 5.0)):
                        inputs = [
                            {
                                "input": (database, "SSA" if country == "ZA" else "World"),
                                "type": "technosphere",
                                "amount": energy,
                            }
                        ]
                        if country == "CA":
                            inputs.append(
                                {
                                    "input": (database, "tailings"),
                                    "type": "technosphere",
                                    "amount": -0.5,
                                }
                            )
                        data[database, f"{metal}_{country}"] = _process(
                            database,
                            f"{metal}_{country}",
                            "platinum group metal, extraction and refinery operations",
                            metal,
                            location=country,
                            exchanges=inputs,
                        )
            bd.Database(database).write(data, searchable=False)
        original = {
            node.key: [dict(row) for row in node.exchanges()] for node in bd.Database("technology")
        }
        ordinary = "ordinary" if year < 2030 else "technology"
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=ordinary,
            technology_database="technology",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache(ordinary, "biosphere"),
        )
        root = bd.get_node(
            database=frontier.database_name, code=provider_service_code("bounded_wind_hydrogen")
        )
        calculation = LCA({root: 1.0})
        calculation.lci()
        assert float(calculation.inventory.sum()) == pytest.approx(
            0.50825 if year < 2030 else 13.505
        )
        assert {
            node.key: [dict(row) for row in node.exchanges()] for node in bd.Database("technology")
        } == original
        copies = {
            node["provider_source_code"]: node
            for node in bd.Database(frontier.database_name)
            if node.get("hybrid_provider_technology_clone")
        }
        if year < 2030:
            assert set(copies) == {"hydrogen", "stack", "iridium", "ruthenium"} | {
                f"{metal}_{country}"
                for metal in ("iridium", "ruthenium")
                for country in ("CA", "ZA", "ZW")
            }
            for code, node in copies.items():
                expected = [
                    {k: v for k, v in row.items() if k != "output"}
                    for row in original["technology", code]
                ]
                for row in expected:
                    child = row["input"][1]
                    if row["type"] == "production":
                        row["input"] = node.key
                    elif child in copies:
                        row["input"] = copies[child].key
                    else:
                        row["input"] = (
                            "ordinary",
                            code.rsplit("_", 1)[1] if child in {"World", "SSA"} else child,
                        )
                assert [
                    {k: v for k, v in dict(row).items() if k != "output"}
                    for row in node.exchanges()
                ] == expected
        else:
            assert not copies
