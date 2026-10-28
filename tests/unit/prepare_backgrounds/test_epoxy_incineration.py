"""Test epoxy municipal treatment and its complete ash chain with synthetic inventories."""

import json
import math
from collections.abc import Iterator
from math import fsum
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from pa_municipal_fixtures import (
    FIXTURE,
    LONG,
    ORIGINS,
    SURFACE,
    URBAN,
    pa_municipal_biosphere,
    pa_municipal_native_inventory,
)

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
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.resin_incineration import (
    _resin_incineration_load,
)


def _native_epoxy_municipal(catalogue: dict[str, Any], database: str, original: bool) -> dict:
    """Reuse complete external functions with a subunit bottom-ash nitrogen release.

    The PHB fixture has no target nitrogen in bottom ash. Actual epoxy does:
    give its source nitrate a 10% short/80% later reference, independent of
    the residual ash's unchanged 10% short/90.2% later source coefficients.
    """
    data = pa_municipal_native_inventory(catalogue, database, original=original)
    nitrate = next(
        key
        for key, node in pa_municipal_biosphere().items()
        if node["name"] == "Nitrate" and tuple(node["categories"]) == LONG
    )
    for row in data[database, "phb_bottom_terminal"]["exchanges"]:
        if row["input"] == nitrate:
            factor = 0.8 / 0.902
            row["amount"] *= factor
            row["loc"] += math.log(factor)
            row["minimum"] *= factor
            row["maximum"] *= factor
    return data


@pytest.fixture(params=[True, False], ids=["original", "future"])
def boundary(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[tuple[Any, _RecipeContext, dict[str, dict[str, object]], str, bool]]:
    """Compile the actual epoxy furnace/ash declarations at two native source dates."""
    original = bool(request.param)
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    kinds = {"epoxy_incineration", "epoxy_ash_market", "epoxy_ash_treatment"}
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["construction"]["kind"] in kinds
    }
    assert len(services) == 15
    names = BrightwayNames(tmp_path / "brightway", "epoxy_municipal", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(pa_municipal_biosphere(), searchable=False)
        bd.Database("original").write(
            _native_epoxy_municipal(catalogue, "original", True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                _native_epoxy_municipal(catalogue, database, False), searchable=False
            )
        before = {
            node.key: [dict(row) for row in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        }
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                cast(str, row["method_authority_id"]): row
                for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        compiled = {
            ("providers", provider_service_code(service_id)): _service_dataset(
                context, service, services, "providers"
            )
            for service_id, service in services.items()
        }
        bd.Database("providers").write(compiled, searchable=False)
        yield bd, context, services, database, original
        assert before == {
            node.key: [dict(row) for row in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        }


def test_epoxy_complete_municipal_function_preserves_hardener_and_native_receiving(boundary):
    """Solve three origin services on principal kg with actual N and signed complete ash."""
    bd, context, services, database, original = boundary
    ratio = 15.44 / 84.56
    principal_carbon = 1635 / 2201
    hardener_carbon = ratio * 96 / 136
    total_carbon = principal_carbon + hardener_carbon
    nitrogen = ratio * 28 / 136
    source_nitrogen = FIXTURE["dry_mass"] * FIXTURE["dry_elements"]["nitrogen"]
    thermal_services = {
        service["construction"]["principal_carbon_origin"]: service
        for service in services.values()
        if service["construction"]["kind"] == "epoxy_incineration"
    }
    assert set(thermal_services) == set(ORIGINS)
    for origin_index, origin in enumerate(ORIGINS):
        service = thermal_services[origin]
        load = _resin_incineration_load(context, service["construction"])
        node = bd.get_node(
            database="providers", code=provider_service_code(service["provider_service_id"])
        )
        assert load.material.wet_mass_kg == pytest.approx(1 + ratio)
        assert load.material.water_kg == 0
        assert load.material.elements_kg["N"] == pytest.approx(nitrogen)
        assert load.material.elements_kg["C"] == pytest.approx(total_carbon)
        assert set(load.material.elements_kg) == {"C", "H", "O", "N"}
        production = node.rp_exchange()
        assert production["amount"] == 1
        properties = production["properties"]
        assert properties["wet mass"]["amount"] == pytest.approx(1 + ratio)
        assert properties["dry mass"]["amount"] == pytest.approx(1 + ratio)
        assert properties["nitrogen content"]["amount"] == pytest.approx(nitrogen / (1 + ratio))
        assert properties["carbon allocation"]["amount"] == pytest.approx(total_carbon)
        owner = json.loads(node["direct_treatment_carbon_owners_json"])[0]
        fractions = {
            (row["flow_database"], row["flow_code"]): row["fraction"]
            for row in owner["principal_carbon_fractions"]
        }
        for name, carbon_factor in (
            ("Carbon dioxide", 12 / 44),
            ("Carbon monoxide", 12 / 28),
            ("Methane", 12 / 16),
        ):
            rows = [row for row in node.biosphere() if row.input["name"].startswith(name + ",")]
            expected_total = load.emissions[name + ", fossil", "kilogram", URBAN]
            assert fsum(row["amount"] for row in rows) == pytest.approx(expected_total)
            fossil = fsum(row["amount"] for row in rows if row.input["name"] == name + ", fossil")
            fossil_share = 1 if origin == "fossil" else hardener_carbon / total_carbon
            assert fossil == pytest.approx(expected_total * fossil_share)
            # Principal-only diagnostics exclude fixed fossil MXDA in every
            # origin; all gas impacts remain in the actual elementary rows.
            principal_gas = fsum(row["amount"] * fractions[row.input.key] for row in rows)
            assert principal_gas * carbon_factor == pytest.approx(
                expected_total * carbon_factor * principal_carbon / total_carbon
            )
            for row in rows:
                assert fsum(row["principal_carbon_origin_fractions"]) == pytest.approx(
                    fractions[row.input.key]
                )
                assert (
                    fsum(
                        share
                        for index, share in enumerate(row["principal_carbon_origin_fractions"])
                        if index != origin_index
                    )
                    == 0
                )
        for name, coefficient, categories in (
            ("Cyanide", 0.000001, URBAN),
            ("Dinitrogen monoxide", 0.00003, URBAN),
            ("Nitrate", 0.000003, SURFACE),
        ):
            row = next(
                row
                for row in node.biosphere()
                if row.input["name"] == name and tuple(row.input["categories"]) == categories
            )
            assert row["amount"] == pytest.approx(coefficient * nitrogen / source_nitrogen)
            assert row["loc"] == pytest.approx(math.log(row["amount"]) + 0.03)
            assert row["scale"] == 0.21
        purchases = {row.input.key: row for row in node.technosphere()}
        for code, share in (
            ("phb_water_a", 0.4 if original else 0.8),
            ("phb_water_b", 0.6 if original else 0.2),
        ):
            row = purchases[database, code]
            assert row["amount"] == pytest.approx(load.duties["water"] * share)
            assert row["loc"] == pytest.approx(math.log(row["amount"]) + 0.03)
            assert row["allocation"] == "invented source allocation"
        assert (database, "phb_haul") not in purchases
        assert (database, "mc_inorganic") not in purchases
        assert purchases[database, "mc_pam"]["amount"] == pytest.approx(load.duties["PAM"])
        lca = LCA({node: 1})
        lca.lci()

        def supply(database_name: str, code: str) -> float:
            """Read the real solved multiplier on the activity's own reference basis."""
            activity = bd.get_node(database=database_name, code=code)
            return float(
                cast(NDArray[np.float64], lca.supply_array)[lca.dicts.activity[activity.id]]
            )

        assert supply("providers", node.key[1]) == pytest.approx(1)
        for fraction, ash in (("bottom", load.bottom_ash), ("residual", load.residual_ash)):
            market_id = service["construction"][fraction + "_ash"]["service_id"]
            terminal_service = next(
                child
                for child in services.values()
                if child["construction"]["kind"] == "epoxy_ash_treatment"
                and child["construction"]["incineration_service_id"]
                == service["provider_service_id"]
                and child["construction"]["ash_fraction"] == fraction
            )
            terminal_code = provider_service_code(terminal_service["provider_service_id"])
            assert purchases["providers", provider_service_code(market_id)][
                "amount"
            ] == pytest.approx(-ash.treatment_mass_kg)
            basis = 3 if original else 7
            assert supply("providers", terminal_code) * basis == pytest.approx(
                ash.treatment_mass_kg, rel=2e-7
            )
            terminal = bd.get_node(database="providers", code=terminal_code)
            emitted_nitrogen = (
                fsum(
                    row["amount"] * 14 / 62
                    for row in terminal.biosphere()
                    if row.input["name"] == "Nitrate"
                )
                / basis
                * ash.treatment_mass_kg
            )
            target_nitrogen = ash.waste_elements_kg["N"] + ash.auxiliary_elements_kg.get("N", 0)
            assert emitted_nitrogen == pytest.approx(
                target_nitrogen * (0.9 if fraction == "bottom" else 1.002)
            )
            assert not any(row.input["name"] == "Zinc II" for row in terminal.biosphere())
        ash_mass = load.bottom_ash.treatment_mass_kg + load.residual_ash.treatment_mass_kg
        for code, coefficient in (("phb_haul", 0.02), ("pa_rail", 0.03), ("pa_barge", 0.027)):
            assert supply(database, code) == pytest.approx(coefficient * ash_mass, rel=2e-7)
        assert supply(database, "phb_hydrated") * (3 if original else 7) == pytest.approx(
            load.residual_ash.treatment_mass_kg, rel=2e-7
        )
        assert load.residual_ash.auxiliary_elements_kg["N"] == pytest.approx(
            load.duties["PAM"] * 14 / 71
        )
        assert load.residual_ash.auxiliary_elements_kg["Ti"] > 0
        assert load.residual_ash.auxiliary_elements_kg["V"] > 0
