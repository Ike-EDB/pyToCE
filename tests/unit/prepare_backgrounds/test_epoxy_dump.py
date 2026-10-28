"""Compile dry cured epoxy at five actual source-function boundaries."""

import json
from math import fsum, log
from typing import Any

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from pa_dump_receiving_fixtures import DUMP_CLASSES, LAND, LAND_ROWS, RURAL, dump_biosphere

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.epoxy_landfill_load import (
    _epoxy_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.resin_dump import _resin_dump_dataset


@pytest.mark.parametrize("future", (False, True), ids=("original", "future"))
def test_complete_dump_binding_preserves_site_basis_and_actual_fixed_hardener(
    tmp_path, future: bool
) -> None:
    """Keep five original climates, dated nonunit land rows and principal-unit disposal."""
    catalogue = load_provider_catalogue()
    services = {s["provider_service_id"]: s for s in catalogue["services"]}
    biosphere = dump_biosphere("biosphere")
    flow_keys = {(row["name"], row["categories"]): key for key, row in biosphere.items()}
    names = BrightwayNames(tmp_path / "brightway", "epoxy_dump", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere, searchable=False)
        for database in ("original", "future") if future else ("original",):
            records = {}
            for i, (_, suffix, precipitation) in enumerate(DUMP_CLASSES):
                service = services[f"epoxy_fossil_open_dump_{suffix}"]
                source = service["construction"]["source"]
                key = database, f"site_{i}"
                p = -(i + (7 if database == "future" else 2))
                date_factor = 1.7 if database == "future" else 1.0
                rows: list[dict[str, Any]] = [{"input": key, "type": "production", "amount": p}]
                for name, _, per_kg in LAND_ROWS:
                    # Two same-flow rows retain unequal shares and independent
                    # uncertainty while normalizing their actual production.
                    for share in (0.2, 0.8):
                        amount = -p * date_factor * per_kg * share
                        rows.append(
                            {
                                "input": flow_keys[name, LAND],
                                "type": "biosphere",
                                "amount": amount,
                                "uncertainty type": 2,
                                "loc": log(amount) + 0.03,
                                "scale": 0.12,
                            }
                        )
                rows.append(
                    {
                        "input": flow_keys["Methane, fossil", RURAL],
                        "type": "biosphere",
                        "amount": 7.0,
                    }
                )
                records[key] = {
                    "name": source["name"],
                    "reference product": source["reference_product"],
                    "unit": source["unit"],
                    "location": source["exact_location"],
                    # A changed future comment cannot resize the original phase reference.
                    "comment": (
                        f"(MAT) of {31 if database == 'future' else 8} C; "
                        f"(MAP) of {1 if database == 'future' else precipitation} mm."
                    ),
                    "exchanges": rows,
                }
            bd.Database(database).write(records, searchable=False)
        database = "future" if future else "original"
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="epoxy_dumps",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                r["method_authority_id"]: r for r in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        compiled = {}
        for origin in ("fossil", "biogenic", "atmospheric_dac"):
            for _, suffix, precipitation in DUMP_CLASSES:
                sid = f"epoxy_{origin}_open_dump_{suffix}"
                data = _resin_dump_dataset(context, services[sid], "epoxy_dumps")
                key = "epoxy_dumps", provider_service_code(sid)
                compiled[key] = data
                rows = data["exchanges"]
                production = next(row for row in rows if row["type"] == "production")
                assert production["amount"] == 1
                assert production["properties"]["wet mass"]["amount"] == pytest.approx(
                    1 + 15.44 / 84.56
                )
                assert production["properties"]["water in wet mass"]["amount"] == 0
                assert not any(row["type"] == "technosphere" for row in rows)
                for name, _, per_kg in LAND_ROWS:
                    land = [row for row in rows if row["input"] == flow_keys[name, LAND]]
                    assert len(land) == 2
                    assert fsum(row["amount"] for row in land) == pytest.approx(
                        per_kg * (1 + 15.44 / 84.56) * (1.7 if future else 1)
                    )
                    for row in land:
                        assert row["loc"] == pytest.approx(log(row["amount"]) + 0.03)
                        assert row["scale"] == 0.12
                actual = _epoxy_landfill_load(
                    origin, _LandfillSite(8, precipitation, 1 - 2.0604 / 7)
                )
                owner = json.loads(data["direct_treatment_carbon_owners_json"])[0]
                fractions = {
                    (r["flow_database"], r["flow_code"]): r["fraction"]
                    for r in owner["principal_carbon_fractions"]
                }
                fossil_gases = [
                    row
                    for row in rows
                    if row.get("name", "") in ("Carbon dioxide, fossil", "Methane, fossil")
                ]
                nonfossil_gases = [
                    row
                    for row in rows
                    if row.get("name", "") in ("Carbon dioxide, non-fossil", "Methane, non-fossil")
                ]
                assert all(
                    fractions[row["input"]]
                    == pytest.approx(
                        actual.carbon[0].phases.input_kg / actual.elements_kg["C"]
                        if origin == "fossil"
                        else 0
                    )
                    for row in fossil_gases
                )
                assert all(fractions[row["input"]] == 1 for row in nonfossil_gases)
                emitted_carbon = fsum(
                    row["amount"] * (12 / 16 if row["name"].startswith("Methane") else 12 / 44)
                    for row in (*fossil_gases, *nonfossil_gases)
                )
                assert emitted_carbon == pytest.approx(actual.phases["C"].gas_kg)
                assert not any(row.get("name") == "Water" for row in rows)
        bd.Database("epoxy_dumps").write(compiled, searchable=False)
        root = bd.get_node(
            database="epoxy_dumps",
            code=provider_service_code(
                "epoxy_biogenic_open_dump_hyperarid_infiltration_class_250mm"
            ),
        )
        lca = LCA({root: 1.0})
        lca.lci()
        assert lca.supply_array[lca.dicts.activity[root.id]] == pytest.approx(1.0)
        # All fifteen functions contain modeled direct discharges and retained native land rows.
        # They contain no material-supply or sanitary-treatment purchases.
        assert len(compiled) == 15
