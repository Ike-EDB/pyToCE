"""Test cured epoxy through its sanitary treatment chain with synthetic inventories."""

import json
from math import fsum, log
from typing import Any, cast

import numpy as np
import pytest
from bw2calc import LCA
from epoxy_landfill_fixtures import OWNERS, biosphere_inventory, native_inventory
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from tps_gas_fixtures import flow_key

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
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.treatment_compilation import (
    attach_landfill_carbon_owners,
)

SURFACE = ("water", "surface water")
LONG = ("water", "ground-, long-term")
RAW = ("water",)
RURAL = ("air", "non-urban air or from high stacks")
ORIGINS = ("fossil", "biogenic", "atmospheric_dac")


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def boundary(request, tmp_path_factory):
    """Compile package declarations against synthetic original and dated inventories."""
    future = request.param
    root = tmp_path_factory.mktemp("epoxy_landfill")
    names = BrightwayNames(root / "brightway", "epoxy_landfill", "biosphere", "original")
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    services = {s["provider_service_id"]: s for s in catalogue["services"]}
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        bd.Database("original").write(native_inventory("original", future=False), searchable=False)
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(native_inventory(database, future=True), searchable=False)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="epoxy_sites",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                cast(str, r["method_authority_id"]): r for r in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        compiled = {}
        for origin in ORIGINS:
            for role in OWNERS:
                sid = f"epoxy_{origin}_{role}"
                compiled["epoxy_sites", provider_service_code(sid)] = _service_dataset(
                    context, services[sid], services, "epoxy_sites"
                )
        attach_landfill_carbon_owners(
            compiled,
            {
                sid: service
                for sid, service in services.items()
                if ("epoxy_sites", provider_service_code(sid)) in compiled
            },
            "epoxy_sites",
        )
        bd.Database("epoxy_sites").write(compiled, searchable=False)
        yield bd, context, database, future


@pytest.mark.parametrize("origin", ORIGINS)
def test_cured_matrix_chain_keeps_signed_native_receivers_and_actual_auxiliary_fates(
    boundary, origin
):
    """Check eleven treatment processes per kg of principal resin and paint treatment quantities."""
    bd, context, database, future = boundary
    prefix = f"epoxy_{origin}_"
    nodes = {
        role: bd.get_node(database="epoxy_sites", code=provider_service_code(prefix + role))
        for role in OWNERS
    }
    site = nodes["landfill"]
    actual = _epoxy_landfill_load(origin, _LandfillSite(16, 1500, 1))
    assert site.rp_exchange()["amount"] == 1.0
    assert site.rp_exchange()["properties"]["wet mass"]["amount"] == pytest.approx(
        1 + 15.44 / 84.56
    )
    lca = LCA({site: 1.0})
    lca.lci()
    assert cast(NDArray[np.float64], lca.supply_array)[
        lca.dicts.activity[site.id]
    ] == pytest.approx(1.0)
    for role, (code, _, _) in OWNERS.items():
        if role != "landfill":
            assert (
                nodes[role].rp_exchange()["amount"]
                == bd.get_node(database=database, code=code).rp_exchange()["amount"]
                < 0
            )
        assert nodes[role]["location"] == "RoW"

    def emitted(role, name, categories):
        """Read native flow quantities per selected activity operation."""
        return fsum(
            row["amount"]
            for row in nodes[role].biosphere()
            if row["input"] == flow_key(name, categories)
        )

    assert emitted("landfill", "TOC, Total Organic Carbon", LONG) == pytest.approx(
        actual.phases["C"].later_leachate_kg
    )
    # Complete site support retains dated functions, actual mass, and individual uncertainty.
    native = bd.get_node(database=database, code="site")
    native_heat = {
        tuple(row.input["categories"]): row
        for row in native.biosphere()
        if row.input["name"] == "Heat, waste"
    }
    assert set(native_heat) == {RURAL, LONG}
    heat_rows = [row for row in site.biosphere() if row.input["name"] == "Heat, waste"]
    assert len(heat_rows) == 3
    assert {tuple(row.input["categories"]) for row in heat_rows} == {
        RURAL,
        LONG,
        ("soil", "industrial"),
    }
    energy = actual.gas.decomposition_trace
    soil_heat = emitted("landfill", "Heat, waste", ("soil", "industrial"))
    later_heat = emitted("landfill", "Heat, waste", LONG)
    assert soil_heat > 0
    assert soil_heat == pytest.approx(energy["gas_allocated_hhv_mj"] - energy["raw_gas_hhv_mj"])
    assert later_heat == pytest.approx(energy["later_leachate_hhv_mj"])
    # Matrix energy is assigned to gas, two ground-heat releases, and short leachate sent to its
    # receiving function.
    assert fsum(
        (soil_heat, later_heat, energy["raw_gas_hhv_mj"], energy["short_leachate_hhv_mj"])
    ) == pytest.approx(energy["input_hhv_mj"])
    site_trace = next(
        json.loads(row["project_terms_json"])
        for row in context.technology_rows
        if row.get("provider_service_id") == prefix + "landfill"
        and row.get("component_id") == "actual_cured_epoxy_landfill"
    )
    admin_kwh = fsum(
        row["amount"]
        for row in site.technosphere()
        if row.input["reference product"] == "electricity, low voltage"
    )
    assert emitted("landfill", "Heat, waste", RURAL) == pytest.approx(
        site_trace["gas"]["heat_air_mj_unbound"] + 3.6 * admin_kwh
    )
    for row in heat_rows:
        compartment = tuple(row.input["categories"])
        if compartment == ("soil", "industrial"):
            assert row["uncertainty type"] == 0
            assert row["loc"] == row["amount"]
            assert "scale" not in row
        elif future:
            source = native_heat[compartment]
            assert row["loc"] == pytest.approx(
                source["loc"] + log(row["amount"] / source["amount"])
            )
            assert row["scale"] == source["scale"]
        else:
            assert row["uncertainty type"] == 0
            assert row["loc"] == row["amount"]
    native_row = next(row for row in native.technosphere() if row.input["code"] == "site_equipment")
    target_row = next(row for row in site.technosphere() if row.input["code"] == "site_equipment")
    multiplier = actual.wet_mass_kg / -native.rp_exchange()["amount"]
    assert target_row["amount"] == pytest.approx(native_row["amount"] * multiplier)
    assert target_row["loc"] == pytest.approx(native_row["loc"] + log(multiplier))
    assert target_row["scale"] == native_row["scale"]
    assert len(list(site.technosphere())) == len(list(native.technosphere()))
    collection = nodes["leachate_collection"]
    plant = nodes["wastewater_treatment"]
    incoming = next(row["amount"] for row in site.technosphere() if row.input == collection)
    original = bd.get_node(database="original", code="site")
    source_liquid = next(
        row["amount"] for row in original.technosphere() if row.input["code"] == "collection"
    )
    assert incoming == pytest.approx(
        source_liquid / -original.rp_exchange()["amount"] * actual.wet_mass_kg
    )
    collected = next(row["amount"] for row in collection.technosphere() if row.input == plant)
    expected_plant = (
        incoming / collection.rp_exchange()["amount"] * collected / plant.rp_exchange()["amount"]
    )
    assert cast(NDArray[np.float64], lca.supply_array)[
        lca.dicts.activity[plant.id]
    ] == pytest.approx(expected_plant, rel=2e-7)
    # Raw bypass and treated outlet indicators have different admitted meanings.
    assert emitted("leachate_collection", "COD, Chemical Oxygen Demand", RAW) == pytest.approx(
        1.09375 * emitted("leachate_collection", "TOC, Total Organic Carbon", RAW)
    )
    assert emitted("leachate_collection", "BOD5, Biological Oxygen Demand", RAW) == pytest.approx(
        0.2589285714 * emitted("leachate_collection", "TOC, Total Organic Carbon", RAW)
    )
    assert emitted("wastewater_treatment", "COD, Chemical Oxygen Demand", SURFACE) == pytest.approx(
        3.7 * emitted("wastewater_treatment", "TOC, Total Organic Carbon", SURFACE)
    )
    assert emitted(
        "wastewater_treatment", "BOD5, Biological Oxygen Demand", SURFACE
    ) == pytest.approx(0.91 * emitted("wastewater_treatment", "TOC, Total Organic Carbon", SURFACE))
    thermal_trace = next(
        row
        for row in context.technology_rows
        if row.get("provider_service_id") == prefix + "sludge_incineration"
        and row.get("component_id") == "actual_sludge_incineration"
    )
    thermal = json.loads(thermal_trace["project_terms_json"])
    bottom = thermal["bottom_ash"]
    # Paint source quench follows oxide-equivalent dry ash.
    dry_bottom = bottom["treatment_mass_kg"] - bottom["water_kg"]
    assert bottom["water_kg"] > 0
    assert bottom["water_kg"] / dry_bottom == pytest.approx(0.186047 / 0.813953)
    assert bottom["water_kg"] / dry_bottom != pytest.approx(0.115628 / 0.884372)
    for name in ("Sodium I", "Calcium II", "Chloride"):
        assert emitted("sludge_incineration", name, SURFACE) > 0
    ash = thermal["residual_ash"]
    normalization = (
        -nodes["residual_ash_treatment"].rp_exchange()["amount"] / ash["treatment_mass_kg"]
    )
    for name, element, short, cumulative in (
        ("Titanium ion", "Ti", 0.0004956, 0.2973),
        ("Vanadium V", "V", 0.002454, 0.7711),
        ("Calcium II", "Ca", 0.0001508, 0.09051),
    ):
        amount = ash["auxiliary_elements_kg"][element]
        assert amount > 0
        assert emitted("residual_ash_treatment", name, SURFACE) == pytest.approx(
            amount * short * normalization
        )
        assert emitted("residual_ash_treatment", name, LONG) == pytest.approx(
            amount * (cumulative - short) * normalization
        )
        assert emitted("bottom_ash_treatment", name, SURFACE) == 0
    # Retain the actual native N release function, including its subunit sum.
    for role, payload in (("bottom_ash_treatment", bottom), ("residual_ash_treatment", ash)):
        source = bd.get_node(database="original", code=OWNERS[role][0])
        properties = source.rp_exchange()["properties"]
        native_n = properties["dry mass"]["amount"] * properties["nitrogen content"]["amount"]
        source_nitrate = (
            fsum(row["amount"] for row in source.biosphere() if row.input["name"] == "Nitrate")
            / -source.rp_exchange()["amount"]
        )
        normalized = (emitted(role, "Nitrate", SURFACE) + emitted(role, "Nitrate", LONG)) / -nodes[
            role
        ].rp_exchange()["amount"]
        assert normalized * payload["treatment_mass_kg"] == pytest.approx(
            source_nitrate * payload["waste_elements_kg"]["N"] / native_n
        )
    # Each carbon owner's signed incoming quantity divided by its production
    # agrees with the solved LCI; fossil hardener/PAM remain outside principal.
    owners = json.loads(site["direct_treatment_carbon_owners_json"])
    assert len(owners) == 4
    for owner in owners:
        node = bd.get_node(database=owner["owner_database"], code=owner["owner_code"])
        assert owner["owner_scale_per_provider_unit"] / node.rp_exchange()[
            "amount"
        ] == pytest.approx(
            cast(NDArray[np.float64], lca.supply_array)[lca.dicts.activity[node.id]], rel=2e-7
        )
        fractions = {
            (r["flow_database"], r["flow_code"]): r["fraction"]
            for r in owner["principal_carbon_fractions"]
        }
        for row in node.biosphere():
            if "principal_carbon_origin_fractions" not in row:
                continue
            assert 0 <= fractions[row["input"]] <= 1
            if origin != "fossil" and row.input["name"].endswith(", fossil"):
                assert fractions[row["input"]] == 0
    field_trace = next(
        json.loads(row["project_terms_json"])
        for row in context.technology_rows
        if row.get("provider_service_id") == prefix + "sludge_landfarming"
        and row.get("project_terms_json")
        and "wet_kg" in row["project_terms_json"]
    )
    field_co2 = fsum(
        emitted("sludge_landfarming", f"Carbon dioxide, {label}", RURAL)
        for label in ("fossil", "non-fossil")
    )
    assert field_co2 / -nodes["sludge_landfarming"].rp_exchange()["amount"] * field_trace[
        "wet_kg"
    ] == pytest.approx(field_trace["actual_elements_kg"]["C"] * 44 / 12)
