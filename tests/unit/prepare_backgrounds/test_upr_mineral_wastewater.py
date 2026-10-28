"""Carry actual UPR minerals through the existing complete water/sludge owners.

The small native-shaped boundary uses invented operating quantities and source
production signs. Its selected mineral-row pattern has positive Al/Co templates
and omits B/Si/Ca. No licensed vector is imported.
"""

import json
from math import fsum, log

import pytest
from material_recipe_fixtures import physical_material_recipe
from sbr_landfill_fixtures import biosphere_inventory, native_inventory
from test_upr_landfill_load import upr_conversion as upr_conversion
from tps_gas_fixtures import flow_key
from tps_water_fixtures import exchange

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    load_provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_treatment import (
    _leachate_market_dataset,
    _wastewater_discharges,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration import (
    _sludge_dataset,
    _sludge_source,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration_load import (
    calculate_sludge_incineration,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_landfarming import (
    _landfarming_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.upr_landfill_load import (
    _upr_landfill_load,
)

SURFACE = ("water", "surface water")
URBAN = ("air", "urban air close to ground")
SOIL = ("soil", "agricultural")
MINERALS = {
    "Boron": "B",
    "Silicon": "Si",
    "Calcium II": "Ca",
    "Aluminium III": "Al",
    "Cobalt II": "Co",
}


def _mineral_biosphere() -> dict:
    """Expose exact accepted flow descriptors with invented test identifiers."""
    data = biosphere_inventory()
    for name in MINERALS:
        for compartment in (SURFACE, URBAN, SOIL):
            data[flow_key(name, compartment)] = {
                "name": name,
                "unit": "kilogram",
                "categories": compartment,
                "type": "emission",
            }
    return data


def _mineral_native(database: str, *, future: bool) -> dict:
    """Retain complete fixture operations with a finite mineral reference pattern."""
    data = native_inventory(database, future=future)
    flows = _mineral_biosphere()
    for code in ("collection", "thermal_sludge", "thermal_landfarming"):
        node = data[database, code]
        node["exchanges"] = [
            row
            for row in node["exchanges"]
            if row["type"] != "biosphere"
            or flows[row["input"]]["name"] not in {"Boron", "Silicon", "Calcium II"}
        ]
        production = next(row for row in node["exchanges"] if row["type"] == "production")
        if code != "collection":
            # An absent source mineral is no fabricated zero assay. Preserve
            # this invented complete dry descriptor by moving its fraction to O.
            for name in ("boron", "silicon"):
                production["properties"]["oxygen content"]["amount"] += production[
                    "properties"
                ].pop(f"{name} content")["amount"]
        for name in ("Aluminium III", "Cobalt II"):
            if code == "collection":
                # Place mixed-plastic raw metal releases in surface water, replacing
                # the generic raw-water compartment inherited from this fixture.
                node["exchanges"] = [
                    row
                    for row in node["exchanges"]
                    if row["type"] != "biosphere" or flows[row["input"]]["name"] != name
                ]
                node["exchanges"].append(
                    exchange(flow_key(name, SURFACE), -production["amount"] * 0.02, "biosphere")
                )
            elif code == "thermal_sludge":
                node["exchanges"].append(
                    exchange(flow_key(name, URBAN), -production["amount"] * 2e-8, "biosphere")
                )
    return data


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def mineral_boundary(request, tmp_path_factory):
    """Resolve dated native APIs for the requested receiving functions."""
    future = request.param
    root = tmp_path_factory.mktemp("upr_mineral_water")
    names = BrightwayNames(root / "brightway", "upr_mineral_water", "biosphere", "original")
    catalogue = load_provider_catalogue()
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(_mineral_biosphere(), searchable=False)
        bd.Database("original").write(_mineral_native("original", future=False), searchable=False)
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(_mineral_native(database, future=True), searchable=False)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="upr_mineral_receiving",
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
        yield (
            bd,
            context,
            database,
            {row["provider_service_id"]: row for row in catalogue["services"]},
        )


@pytest.mark.parametrize("origin", ("fossil", "biogenic", "atmospheric_dac"))
def test_upr_minerals_reach_water_soil_and_thermal_phases_once(
    mineral_boundary, upr_conversion, origin
):
    """Preserve source removal, positive endpoints, fresh calcium and dated supplies."""
    bd, context, database, services = mineral_boundary
    load = _upr_landfill_load(
        upr_conversion,
        origin,
        _LandfillSite(16.743579766537, 1250.84108527132, 1.0),
        carbonate_end_year=2863,
    )
    # The original mixed-plastic hydraulic reference is m3/kg whole waste. This test starts from
    # actual mineral-containing leachate; phase chemistry is tested separately.
    volume = 0.00395372891508696 * load.wet_mass_kg
    water = _wastewater_load(
        load.leachate,
        volume_m3=volume,
        incoming_water_kg=1000 * volume - fsum(load.leachate.elements_kg.values()),
        evaporation_fraction=0.0,
        profile=_ROW_WASTEWATER_PROFILE,
    )
    # Original avgWWT D/N rows 27/55/58/57/36: primary and primary+secondary
    # removal, independently listed from the implementation. No mineral gas.
    source_removal = {
        "B": (0.024019502621814723, 0.080065008739382409),
        "Si": (0.16769428387925497, 0.55898094626418326),
        "Al": (0.13960078417394403, 0.46533594724648014),
        "Ca": (0.062112441519308538, 0.20704147173102846),
        "Co": (0.25770080368472997, 0.85900267894909987),
    }
    f1, f2, f3 = _ROW_WASTEWATER_PROFILE.hydraulic.stages
    for element, (primary, all_removal) in source_removal.items():
        incoming = load.leachate.elements_kg[element]
        assert incoming > 0
        solids = incoming * (primary * f1 + all_removal * (f2 + f3))
        assert water.primary_sludge.elements_kg[element] == pytest.approx(incoming * primary)
        assert water.final_sludge.elements_kg[element] == pytest.approx(solids, rel=2e-13)
        assert water.effluent.elements_kg[element] == pytest.approx(incoming - solids, rel=2e-13)
        assert (
            water.biological_air_elements_kg[element]
            == water.digester_gas_elements_kg[element]
            == 0
        )
    discharges = _wastewater_discharges(
        water,
        _ROW_WASTEWATER_PROFILE,
        gross_electricity_kwh=water.duties.electricity_kwh,
        gas_heat_air_mj=0.0,
        purchased_heat_mj=water.duties.heat_mj,
    )
    for name, element in MINERALS.items():
        effluent_rows = [row for row in discharges if row.name == name]
        assert len(effluent_rows) == 1
        assert effluent_rows[0].categories == SURFACE
        assert effluent_rows[0].amount == pytest.approx(water.effluent.elements_kg[element])
    assert water.effluent.cod_kg_o2 == pytest.approx(3.7 * water.effluent.toc_kg)
    assert water.effluent.bod5_kg_o2 == pytest.approx(0.91 * water.effluent.toc_kg)
    assert water.duties.polyacrylamide_kg > 0
    assert water.water.effluent_kg > 0

    # Shared receiver declarations exercise their direct source-binding
    # boundary with the UPR material. Root UPR route compilation and final
    # ash identities are tested at their owners.
    prefix = f"sbr_filled_{origin}_"
    thermal_service = services[prefix + "sludge_incineration"]
    construction = thermal_service["construction"]
    source = _sludge_source(context, construction)
    thermal = calculate_sludge_incineration(
        source,
        water,
        "actual_upr",
        source_carbon_origins=("fossil",),
        additional_apc_water_flows=tuple(construction["additional_apc_water_flows"]),
        phosphorus_phase_reference=False,
        bottom_ash_water_to_dry_ratio=0.197802 / 0.802198,
    )
    assert all(element not in source.elements_kg for element in ("B", "Si", "Ca"))
    assert thermal.thermal["calorific_organic_oxygen_kg"] < thermal.allocated_oxygen_kg
    assert thermal.bottom_ash.water_kg == pytest.approx(
        (thermal.bottom_ash.treatment_mass_kg - thermal.bottom_ash.water_kg) * 0.197802 / 0.802198
    )
    # Original sludgeIncin K/L rows 55/57 give elemental fractions. Fresh APC calcium has its own
    # exit.
    for name, element, fraction in (
        ("Silicon", "Si", 0.0000015575195263856684),
        ("Calcium II", "Ca", 0.0078698725106003327),
    ):
        incoming = thermal.elements_kg[element]
        fresh = thermal.aqueous_apc_elements_kg["Ca"] if element == "Ca" else 0.0
        assert thermal.emissions[name, "kilogram", URBAN] == pytest.approx(incoming * fraction)
        assert thermal.emissions[name, "kilogram", SURFACE] == pytest.approx(
            incoming * fraction + fresh
        )
    assert thermal.aqueous_apc_elements_kg["Ca"] > 0
    assert not any(key[0] == "Boron" and amount for key, amount in thermal.emissions.items())
    b = thermal.elements_kg["B"]
    assert thermal.bottom_ash.waste_elements_kg["B"] == pytest.approx(b * 0.87871212878712129)
    assert thermal.residual_ash.waste_elements_kg["B"] == pytest.approx(
        b * (0.030734214714121816 + 0.090153696494757313 + 0.00039996000399960013)
    )

    market = _leachate_market_dataset(
        context,
        services[prefix + "leachate_collection"],
        load.leachate,
        water,
        "upr_mineral_receiving",
    )
    furnace = _sludge_dataset(context, thermal_service, thermal, "upr_mineral_receiving")
    field = _landfarming_dataset(
        context,
        services[prefix + "sludge_landfarming"],
        water,
        "actual_upr",
        "upr_mineral_receiving",
        complete_carbon_oxidation=True,
        phosphorus_equilibrium=False,
    )

    def rows(dataset, name, compartment):
        """Read compiled exchanges by exact native identity and compartment."""
        return [
            row
            for row in dataset["exchanges"]
            if row["type"] == "biosphere" and row["input"] == flow_key(name, compartment)
        ]

    def production(dataset):
        """Read the unchanged signed native output denominator."""
        return next(row["amount"] for row in dataset["exchanges"] if row["type"] == "production")

    native_market = bd.get_node(database=database, code="collection")
    native_plant = next(row for row in native_market.technosphere() if row.input["code"] == "plant")
    bypass = 1 - native_plant["amount"] / native_market.rp_exchange()["amount"]
    for name, element in MINERALS.items():
        raw = rows(market, name, SURFACE)
        assert len(raw) == 1
        assert raw[0]["amount"] / -production(market) * volume == pytest.approx(
            bypass * load.leachate.elements_kg[element]
        )
        soil = rows(field, name, SOIL)
        assert len(soil) == 1
        assert soil[0]["amount"] / -production(
            field
        ) * water.landfarmed_sludge_wet_kg == pytest.approx(
            water.landfarming_dry_share * water.final_sludge.elements_kg[element]
        )
        for compartment in (URBAN, SURFACE):
            emitted = rows(furnace, name, compartment)
            if element == "B":
                assert emitted == []
                continue
            assert len(emitted) == 1
            assert emitted[0]["amount"] / -production(
                furnace
            ) * thermal.wet_mass_kg == pytest.approx(
                thermal.emissions[name, "kilogram", compartment]
            )
            if element in {"Al", "Co"}:
                original = bd.get_node(database="original", code="thermal_sludge")
                template = next(
                    row
                    for row in original.biosphere()
                    if row.input.key == flow_key(name, compartment)
                )
                assert emitted[0]["scale"] == template["scale"]
                assert emitted[0]["loc"] == pytest.approx(
                    template["loc"] + log(emitted[0]["amount"] / template["amount"])
                )
            else:
                assert emitted[0]["uncertainty type"] == 0
                assert "scale" not in emitted[0]
        if element in {"B", "Si", "Ca"}:
            assert soil[0]["uncertainty type"] == raw[0]["uncertainty type"] == 0

    # Complete original/future spreading and the independent hydraulic market
    # support are unchanged. No mineral procurement or duplicate factory exists.
    for dataset, code, replaced in (
        (field, "thermal_landfarming", None),
        (market, "collection", "plant"),
    ):
        expected = [
            row
            for row in bd.get_node(database=database, code=code).technosphere()
            if row.input["code"] != replaced
        ]
        actual = [
            row
            for row in dataset["exchanges"]
            if row["type"] == "technosphere" and (replaced is None or row["input"][0] == database)
        ]
        assert [(row["input"], row["amount"]) for row in actual] == [
            (row.input.key, row["amount"]) for row in expected
        ]
    farm_trace = next(
        json.loads(row["project_terms_json"])
        for row in reversed(context.technology_rows)
        if row.get("component_id") == "actual_sludge_landfarming"
    )
    assert set(farm_trace["additional_mineral_soil_kg"]) == {"Boron", "Silicon", "Calcium II"}
    assert farm_trace["actual_elements_kg"]["Al"] > 0
    assert {pool["component"] for pool in farm_trace["carbon_components"]} >= {
        "principal_upr",
        "upr_principal_reference_difference",
        "upr_retained_auxiliaries",
        "wastewater_polyacrylamide",
    }
