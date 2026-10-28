"""Check PBS finished yield, delivery and linked municipal incineration and ash treatment.

Purchased reactor feed processes supply one unit; their inventories are synthetic.
Existing equations determine reactor quantities, hydraulic collection, finishing,
delivery and treatment.
"""

import json
from math import fsum
from typing import Any, cast

import numpy as np
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from pa_municipal_fixtures import ORIGINS, _biosphere, _native

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
    _route_contexts,
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.resin_incineration import (
    _resin_incineration_load,
)


@pytest.fixture(scope="module", params=(True, False), ids=("original", "future"))
def boundary(request, tmp_path_factory):
    """Compile finished PBS products with synthetic original and future suppliers."""
    original = request.param
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    feeds = {
        component["service_id"]
        for origin in ORIGINS
        for component in services[f"pbs_reactor_product_{origin}"]["construction"]["components"]
        if "service_id" in component
        and component["service_id"] != "hydraulic_water_collection_europe"
    }
    # Native manufacture of these purchased services is outside this test's
    # finishing boundary. Their actual recipe quantities/signs remain untouched.
    for sid in feeds:
        services[sid]["construction"] = {
            "kind": "native_activity",
            "source_role": "native_ecoinvent_provider",
            "amount": 1.0,
            "selector": {
                "name": f"invented PBS supply {sid}",
                "reference_product": sid,
                "unit": services[sid]["unit"],
                "exact_location": "RoW",
            },
        }
    selected = {f"pbs_{origin}" for origin in ORIGINS} | {
        "recycling_pet_pellet_delivery",
        "hydraulic_water_collection_europe",
        "hydraulic_water_treatment_europe",
    }
    selected |= {sid for sid in services if sid.startswith("pbs_") and "_finishing_reject_" in sid}
    names = BrightwayNames(
        tmp_path_factory.mktemp("pbs_finish") / "brightway", "pbs_finish", "biosphere", "original"
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(_biosphere(), searchable=False)
        utility_codes = {}
        for database, source_date in (("original", True), ("future", False)):
            if original and not source_date:
                continue
            data = _native(catalogue, database, original=source_date)

            def supply(code, selector, *, production=1.0, exchanges=()):
                """Write a complete invented supplier at its independent production basis."""
                data[database, code] = {
                    "name": selector["name"],
                    "reference product": selector["reference_product"],
                    "unit": selector["unit"],
                    "location": selector["exact_location"],
                    "type": "process",
                    "exchanges": [
                        {"input": (database, code), "type": "production", "amount": production},
                        *exchanges,
                    ],
                }

            for sid in feeds:
                supply(
                    sid,
                    services[sid]["construction"]["selector"],
                )
            components = services["pbs_reactor_product_fossil"]["construction"]["components"]
            for component in components:
                if "selector" not in component:
                    continue
                selector = component["selector"]
                key = next(
                    (
                        key
                        for key, row in data.items()
                        if (row["name"], row["reference product"], row["unit"], row["location"])
                        == (
                            selector["name"],
                            selector["reference_product"],
                            selector["unit"],
                            selector["exact_location"],
                        )
                    ),
                    None,
                )
                if key is None:
                    key = (database, f"pbs_{component['component_id']}")
                    supply(key[1], selector)
                utility_codes[component["component_id"]] = key[1]
            delivery = services["recycling_pet_pellet_delivery"]["construction"]
            principal = delivery["omit_technosphere"][0]["selector"]
            supply("pbs_unused_pet", {**principal, "exact_location": principal["location"]})
            supply(
                "pbs_delivery_haul",
                {
                    "name": "invented final pellet freight",
                    "reference_product": "freight",
                    "unit": "ton kilometer",
                    "exact_location": "RER",
                },
            )
            basis = 3.0 if source_date else 7.0
            freight = 0.25 if source_date else 0.5
            supply(
                "pbs_delivery",
                delivery["base"]["selector"],
                production=basis,
                exchanges=(
                    {
                        "input": (database, "pbs_unused_pet"),
                        "type": "technosphere",
                        "amount": basis,
                    },
                    {
                        "input": (database, "pbs_delivery_haul"),
                        "type": "technosphere",
                        "amount": basis * freight,
                    },
                ),
            )
            bd.Database(database).write(data, searchable=False)
        database = "original" if original else "future"
        before = {
            node.key: [dict(ex) for ex in node.exchanges()]
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
        bd.Database("providers").write(
            {
                ("providers", provider_service_code(sid)): _service_dataset(
                    context, services[sid], services, "providers"
                )
                for sid in sorted(selected)
            },
            searchable=False,
        )
        yield bd, context, services, database, original, utility_codes
        assert before == {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        }


def test_finished_output_scales_reactor_once_and_delivers_only_final_kg(boundary):
    """Separate gross reactor, finished delivery and one complete reject receiver."""
    bd, context, services, database, original, utilities = boundary
    loss = (1 - 0.98) / 0.98
    for origin in ORIGINS:
        root = bd.get_node(database="providers", code=provider_service_code(f"pbs_{origin}"))
        assert root.rp_exchange()["amount"] == 1
        purchases = {ex.input.key: ex["amount"] for ex in root.technosphere()}
        for sid, coefficient in (
            (f"succinic_acid_delivered_{origin}", 0.6849009345381919),
            (f"butanediol_delivered_{origin}", 0.5227003759301838),
            ("titanium_butoxide_delivered", 0.001369801869076384),
        ):
            assert purchases[database, sid] == pytest.approx(coefficient / 0.98, rel=2e-15)
        water_code = provider_service_code("hydraulic_water_collection_europe")
        assert purchases["providers", water_code] == pytest.approx(
            -0.0002089711123374521 / 0.98, rel=2e-15
        )
        assert purchases[database, utilities["electricity"]] == pytest.approx(
            0.07 / 0.98 + 1.1 / (3.6 * 0.98), rel=2e-15
        )
        for name, amount in (
            ("process_water", 0.49),
            ("cooling_water", 0.26),
            ("low_pressure_steam", 0.09),
        ):
            assert purchases[database, utilities[name]] == pytest.approx(amount / 0.98, rel=2e-15)
        receiver_id = f"pbs_{origin}_finishing_reject_incineration"
        receiver = bd.get_node(database="providers", code=provider_service_code(receiver_id))
        assert purchases[receiver.key] == loss
        assert purchases["providers", provider_service_code("recycling_pet_pellet_delivery")] == 1
        assert len(purchases) == 10
        lca = LCA({root: 1})
        lca.lci()

        def solved(db, code):
            """Return a real LCI supply multiplier for the exact intended endpoint."""
            node = bd.get_node(database=db, code=code)
            return float(cast(NDArray[np.float64], lca.supply_array)[lca.dicts.activity[node.id]])

        assert solved(database, "pbs_unused_pet") == 0
        assert solved(database, "pbs_delivery_haul") == pytest.approx(0.25 if original else 0.5)
        assert solved("providers", receiver.key[1]) == pytest.approx(loss, rel=2e-7)
        load = _resin_incineration_load(context, services[receiver_id]["construction"])
        for fraction, ash in (("bottom", load.bottom_ash), ("residual", load.residual_ash)):
            code = provider_service_code(f"pbs_{origin}_finishing_reject_{fraction}_ash_treatment")
            assert solved("providers", code) * (3 if original else 7) == pytest.approx(
                loss * ash.treatment_mass_kg, rel=2e-7
            )
        assert solved("providers", water_code) == pytest.approx(
            0.0002089711123374521 / 0.98 / 2, rel=2e-7
        )


def test_reject_preserves_finished_composition_and_fossil_catalyst(boundary):
    """Keep one kg nominal dry PBS and two carbon pools in every dated receiver."""
    bd, context, services, database, original, utilities = boundary
    q, residue = 0.9986301981309236, 0.001369801869076384
    for origin in ORIGINS:
        sid = f"pbs_{origin}_finishing_reject_incineration"
        load = _resin_incineration_load(context, services[sid]["construction"])
        assert load.material.wet_mass_kg == 1
        assert load.material.water_kg == 0
        assert load.material.elements_kg["Ti"] == pytest.approx(residue * 48 / 340, rel=2e-15)
        assert fsum(load.material.elements_kg.values()) == pytest.approx(1, rel=2e-15)
        assert [(pool.origin, pool.principal, pool.carbon_kg) for pool in load.material.carbon] == [
            (origin, True, pytest.approx(q * 96.088 / 172.180, rel=2e-15)),
            ("fossil", False, pytest.approx(residue * 192 / 340, rel=2e-15)),
        ]
        receiver = bd.get_node(database="providers", code=provider_service_code(sid))
        production = receiver.rp_exchange()
        assert production["amount"] == 1
        properties = production["properties"]
        assert properties["carbon content, fossil"]["amount"] == pytest.approx(
            residue * 192 / 340 + (q * 96.088 / 172.180 if origin == "fossil" else 0)
        )
        assert properties["titanium content"]["amount"] == pytest.approx(residue * 48 / 340)
        assert not {
            (database, "titanium_butoxide_delivered"),
            ("providers", provider_service_code(f"pbs_{origin}")),
        } & {ex.input.key for ex in receiver.technosphere()}
        owners = json.loads(receiver["direct_treatment_carbon_owners_json"])
        assert owners[0]["origin_mapping_status"] == "actual_pbs_principal_origin"
        assert owners[0]["owner_scale_per_provider_unit"] == 1


def test_finished_delivery_and_new_receivers_retain_exact_route_ownership():
    """Deliver six final resin roots once while PBS alone requires finishing."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    contexts = _route_contexts(catalogue["routes"], services)
    for origin in ORIGINS:
        for polymer in ("pbs", "pbat"):
            components = services[f"{polymer}_{origin}"]["construction"]["components"]
            assert [
                row
                for row in components
                if row.get("service_id") == "recycling_pet_pellet_delivery"
            ] == [
                {
                    "component_id": "finished_resin_delivery",
                    "service_id": "recycling_pet_pellet_delivery",
                    "amount": 1.0,
                }
            ]
            if polymer == "pbat":
                assert not any("finishing" in row.get("component_id", "") for row in components)
        receiver = f"pbs_{origin}_finishing_reject_incineration"
        assert {row["provider_demand_type"] for row in contexts[receiver]} == {
            "virgin_resin_production"
        }
        assert {
            row["provider_service_id"]
            for row in catalogue["routes"]
            if row["route_id"] == f"bioplastics_incineration__pbs__{origin}"
        } == {"incineration_pet" if origin == "fossil" else "incineration_pet__non_fossil"}
        assert (
            services[f"pbs_{origin}"]["representative_method"]["titanium_butoxide_kg"]
            == 0.001369801869076384
        )
