"""Route polyester reaction water once through complete signed hydraulic services.

These tests join six manufacturing component declarations to the wastewater
functions. The synthetic graph has invented quantities and nonunit treatment
outputs; separate tests cover the hydraulic equations.
"""

import json
from importlib.resources import files
from math import fsum
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from water_treatment_fixtures import water_biosphere, water_native_inventory

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_direct_source_ids,
    _service_source_closures,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _flatten_service,
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
_COLLECTION = "hydraulic_water_collection_europe"
_TREATMENT = "hydraulic_water_treatment_europe"
# Per kg PBS reactor product or finished PBAT, nominal H2O after the
# separately owned organic recovery/purge. PBS finishing scales its reactor
# basis once. Conversion to m3 uses the admitted 1000 kg/m3 water basis.
_REACTION_WATER_KG = {"pbs": 0.2089711123374521, "pbat": 0.17212497471760035}


@pytest.fixture(scope="module")
def polyester_catalogues() -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the packaged declarations and the actual augmented provider catalog."""
    raw = json.loads(
        files("pytoce.prepare_backgrounds.data.provider_recipes")
        .joinpath("provider_catalogue.json")
        .read_text(encoding="utf-8")
    )
    return raw, load_provider_catalogue()


@pytest.mark.parametrize("polymer", ("pbs", "pbat"))
@pytest.mark.parametrize("origin", _ORIGINS)
def test_polyester_reaction_water_has_one_source_bound_receiver(
    polyester_catalogues: tuple[dict[str, Any], dict[str, Any]], polymer: str, origin: str
) -> None:
    """Keep formation water distinct from fresh water, recovered BDO and THF."""
    service_id = f"{polymer}_{origin}"
    # Augmentation supplies declared source authorities before closure admission;
    # the raw resource still independently fixes all six component quantities.
    closures = _service_source_closures(polyester_catalogues[1])
    for catalogue in polyester_catalogues:
        services = {row["provider_service_id"]: row for row in catalogue["services"]}
        service = services[f"pbs_reactor_product_{origin}" if polymer == "pbs" else service_id]
        construction = service["construction"]
        components = construction["components"]
        reaction = [
            row for row in components if row.get("component_id") == "reaction_water_treatment"
        ]
        assert reaction == [
            {
                "component_id": "reaction_water_treatment",
                "service_id": _COLLECTION,
                "amount": -_REACTION_WATER_KG[polymer] / 1000,
            }
        ]
        # Demand collection once, with no whole-sewage or separate plant purchase.
        assert [row for row in components if row.get("service_id") == _COLLECTION] == reaction
        assert not any(
            row.get("selector", {}).get("reference_product") == "wastewater, average"
            or row.get("service_id") == _TREATMENT
            for row in components
        )
        assert services[_COLLECTION]["unit"] == "cubic meter"
        outputs = {row["source_term"]: row for row in construction["source_balance_outputs"]}
        assert outputs["polycondensation reaction water"]["unit"] == "kilogram"
        assert outputs["polycondensation reaction water"]["amount"] == _REACTION_WATER_KG[polymer]
        amounts = {
            row.get("component_id", row.get("service_id")): row["amount"] for row in components
        }
        if polymer == "pbs":
            assert (amounts["process_water"], amounts["cooling_water"]) == (0.49, 0.26)
            assert outputs["internal recovered BDO"]["amount"] == 0.1563681587988443
            assert amounts[f"butanediol_delivered_{origin}"] == 0.5227003759301838
            assert set(outputs) == {"internal recovered BDO", "polycondensation reaction water"}
        else:
            assert amounts["cooling_water"] == 7.09830384
            assert amounts["tetrahydrofuran_purge_treatment"] == -0.00944
            assert outputs["tetrahydrofuran purge"]["amount"] == 0.00944
            assert set(outputs) == {"tetrahydrofuran purge", "polycondensation reaction water"}
        direct = _service_direct_source_ids(catalogue, service)
        assert "eu_pol_bref_2007" in direct
        assert "doka_2021_wastewater_carbon_transfers" in closures[_COLLECTION]
        assert set(closures[_COLLECTION]) <= set(closures[service_id])


@pytest.mark.parametrize("original", (True, False), ids=("original", "future"))
def test_polyester_components_solve_the_complete_dated_water_chain(
    tmp_path: Path, polyester_catalogues: tuple[dict[str, Any], dict[str, Any]], original: bool
) -> None:
    """Flatten actual component demands, then solve negative native outputs once.

    Native collection produces -2 m3 and buys -0.5 m3 from a plant producing
    -4 m3. Consequently V m3 gives V/2 collection and V/16 plant operations.
    This checks the manufacturing link, native sign, and production references together.
    """
    catalogue = polyester_catalogues[1]
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    names = BrightwayNames(tmp_path / "brightway", "polyester_water", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(water_biosphere(), searchable=False)
        bd.Database("original").write(
            water_native_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database("future").write(
                water_native_inventory(catalogue, "future", original=False), searchable=False
            )
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
                str(row["method_authority_id"]): row for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        bd.Database("providers").write(
            {
                ("providers", provider_service_code(sid)): _service_dataset(
                    context, services[sid], services, "providers"
                )
                for sid in (_COLLECTION, _TREATMENT)
            },
            searchable=False,
        )
        exchanges: dict[tuple[tuple[str, str], str], float] = {}
        for polymer in _REACTION_WATER_KG:
            for origin in _ORIGINS:
                service_id = f"{polymer}_{origin}"
                scale = 1.0
                if polymer == "pbs":
                    reactor = services[service_id]["construction"]["components"][0]
                    service_id = reactor["service_id"]
                    scale = reactor["amount"]
                components = services[service_id]["construction"]["components"]
                component = next(
                    row
                    for row in components
                    if row.get("component_id") == "reaction_water_treatment"
                )
                _flatten_service(
                    context,
                    component["service_id"],
                    scale=scale * component["amount"],
                    services=services,
                    exchanges=exchanges,
                )
        volume = 3 * fsum((_REACTION_WATER_KG["pbs"] / 0.98, _REACTION_WATER_KG["pbat"])) / 1000
        market = bd.get_node(database="providers", code=provider_service_code(_COLLECTION))
        plant = bd.get_node(database="providers", code=provider_service_code(_TREATMENT))
        assert exchanges == {(market.key, "technosphere"): pytest.approx(-volume, rel=2e-15)}
        assert market.rp_exchange()["amount"] == -2
        assert plant.rp_exchange()["amount"] == -4
        assert {ex.input.key: ex["amount"] for ex in market.technosphere()} == {
            plant.key: -0.5,
            (database, "sewer_rural"): 0.02,
            (database, "sewer_urban"): 0.04,
        }
        support = {ex.input.key: ex["amount"] for ex in plant.technosphere()}
        assert {
            code: support[database, code] for code in ("limestone", "plant", "screenings", "sand")
        } == {"limestone": 0.008, "plant": 0.002, "screenings": -0.004, "sand": 0.006}
        assert support[database, "heat_europe"] > 0
        assert support[database, "electricity_a"] / support[
            database, "electricity_b"
        ] == pytest.approx(1 / 3 if original else 3)
        lca = LCA({market: exchanges[market.key, "technosphere"]})
        lca.lci()
        supply = cast(Any, lca.supply_array)
        assert supply[lca.dicts.activity[market.id]] == pytest.approx(volume / 2, rel=2e-15)
        assert supply[lca.dicts.activity[plant.id]] == pytest.approx(volume / 16, rel=2e-15)
        inventory = cast(Any, lca.inventory)
        water_ids = [
            bd.get_node(database="biosphere", code=code).id
            for code in ("surface", "water_urban", "water_rural")
        ]
        # The installed public Brightway writer serializes matrix amounts as
        # float32. Bound this solved sum by that precision; component quantities
        # and signed production/supply assertions above retain their own basis.
        assert fsum(
            float(inventory[lca.dicts.biosphere[i], :].sum()) for i in water_ids
        ) == pytest.approx(volume, rel=2**-23, abs=0)
        # Formation water carries no sewage carbon/COD. The real supplies still
        # emit carbon upstream; the receiving link must preserve those burdens.
        for node in (market, plant):
            assert not any(ex.input.key[1] in {"cod", "co2"} for ex in node.biosphere())
        co2 = bd.get_node(database="biosphere", code="co2")
        assert float(inventory[lca.dicts.biosphere[co2.id], :].sum()) > 0
