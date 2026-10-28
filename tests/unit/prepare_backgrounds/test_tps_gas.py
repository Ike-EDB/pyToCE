"""Test actual pyToCE TPS gas with compact invented Brightway sources.

The real three formulations feed the real residue and conditional wastewater
owners. These tests stop at the gas-owner boundary; site collection, biological
emissions and whole-plant wastewater duties are separate integration contracts.
No private source generator, licensed database or package monkeypatch is used.

Direct equations use relative 2e-12, absolute zero tolerance. Small linked LCI
checks use 3e-7 relative tolerance because bw_processing's external exchange
array stores float32 amounts, including several linked nonunit coefficients.
"""

from contextlib import contextmanager
from copy import deepcopy
from math import fsum, log
from typing import Any, cast

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from tps_gas_fixtures import (
    CHP_FUEL,
    CHP_LHV,
    CHP_PRODUCTION,
    ENGINE_FACTORS,
    FURNACE_FUEL,
    FURNACE_LHV,
    FURNACE_N2O,
    FURNACE_PRODUCTION,
    GAS_SOURCES,
    LOG_OFFSET,
    LOG_SPREAD,
    NMVOC,
    ORIGINS,
    RURAL,
    URBAN,
    biosphere_inventory,
    exchange,
    native_inventory,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    load_provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.tps_residue_load import (
    _tps_landfill_gas_load,
    _tps_residue_load,
)
from pytoce.prepare_backgrounds.preparation.providers.waste_gas import (
    _landfill_gas,
    _wastewater_gas,
)


@pytest.fixture(scope="module")
def gas_loads():
    """Prepare actual formulations once on one kg principal, before collection share."""
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    loads = {}
    for origin in ORIGINS:
        residue = _tps_residue_load(services, f"tps_{origin}")
        # Synthetic inputs supply 1.8 kg carrier water, 2 L wastewater, and 1% climatic evaporation
        # once.
        water = _wastewater_load(
            residue.leachate,
            volume_m3=0.002,
            incoming_water_kg=1.8,
            evaporation_fraction=0.01,
            profile=_ROW_WASTEWATER_PROFILE,
        )
        loads[origin] = residue, water
    return catalogue, loads


@contextmanager
def _gas_context(root, catalogue, *, future):
    """Write fresh source stores through public Brightway APIs in a disposable project."""
    names = BrightwayNames(root / "brightway", "tps_gas", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        bd.Database("original").write(
            native_inventory("original", future=False),
            searchable=False,
        )
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(native_inventory(database, future=True), searchable=False)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                row["method_authority_id"]: row for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        yield bd, context, database, int(future)


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def gas_boundary(request, tmp_path_factory, gas_loads):
    """Reuse each immutable invented original/future graph across gas assertions."""
    catalogue, _ = gas_loads
    with _gas_context(tmp_path_factory.mktemp("tps_gas"), catalogue, future=request.param) as case:
        yield case


def _compile(context, stage, loads):
    """Call the actual gas owner with its admitted construction and typed load."""
    construction = {"gas_sources": deepcopy(GAS_SOURCES)}
    if stage == "landfill":
        return _landfill_gas(
            context, construction, _tps_landfill_gas_load(loads[0]), "tps_landfill_test"
        )
    return _wastewater_gas(
        context, construction, loads[1], "tps_wastewater_test", profile=_ROW_WASTEWATER_PROFILE
    )


def _inputs(stage, loads):
    """State the independent gas-inlet contract from the actual upstream owners.

    Doka2003 RoW collection: .47 bypass, .1802 flare, .3498 CHP. Doka2021
    digester: 65/35 methane/CO2, .0075 homogeneous leak, GNIWWT CHP share
    .6045658030869254; avgWWT AK10 supplies NH3-N. Carbon quantities are kg C.
    """
    residue, water = loads
    if stage == "landfill":
        pools = {
            (x.component, x.principal, x.origin): (x.methane_carbon_kg, x.carbon_dioxide_carbon_kg)
            for x in residue.carbon
        }
        return (
            pools,
            residue.protein_nitrogen.gas_kg,
            residue.protein_sulfur.gas_kg,
            0.0,
            (
                0.47,
                0.1802,
                0.3498,
            ),
        )
    pools = {
        (x.component, x.principal, x.origin): (
            0.65 * x.digester_gas_carbon_kg,
            0.35 * x.digester_gas_carbon_kg,
        )
        for x in water.carbon
    }
    n = water.digester_gas_elements_kg["N"]
    return (
        pools,
        n,
        water.digester_gas_elements_kg["S"],
        0.015136275611828884 * n,
        (
            0.0075,
            0.9925 * (1 - 0.6045658030869254),
            0.9925 * 0.6045658030869254,
        ),
    )


def _emissions(bd, gas):
    """Sum returned rows by exact native name/compartment, preserving repeated origins."""
    grouped = {}
    for row in gas.biosphere_exchanges:
        flow = bd.get_node(database=row["input"][0], code=row["input"][1])
        key = flow["name"], tuple(flow["categories"])
        grouped.setdefault(key, []).append(row["amount"])
    return {key: fsum(values) for key, values in grouped.items()}


@pytest.mark.parametrize("stage", ("landfill", "wastewater"))
@pytest.mark.parametrize("origin", ORIGINS)
def test_gas_species_elements_and_carbon_origins(gas_boundary, gas_loads, stage, origin):
    """Conserve each inlet carbon pool and actual gas H/N/S."""
    bd, context, _, date = gas_boundary
    loads = gas_loads[1][origin]
    gas = _compile(context, stage, loads)
    if stage == "landfill":
        assert gas.heat_soil_mj >= 0
    pools, nitrogen, sulfur, ammonia_n, fractions = _inputs(stage, loads)
    leak, flare, chp = fractions
    emissions = _emissions(bd, gas)
    urban = 0.0 if stage == "landfill" else 0.6808092816046385  # GNIWWT AY337 gas split
    seen_origins = {key[2] for key, masses in pools.items() if fsum(masses) > 0}
    assert seen_origins <= {"biogenic", "fossil", "atmospheric_dac"}
    if stage == "wastewater":
        assert any(
            key[0] == "wastewater_polyacrylamide" and not key[1] and key[2] == "fossil"
            for key in pools
        )
    if stage == "landfill" and origin == "biogenic":
        assert nitrogen == sulfur == 0
        assert not any("fossil" == key[2] for key in pools)
        assert not any(
            name in {"Ammonia", "Sulfur dioxide", "Carbon dioxide, fossil", "Methane, fossil"}
            for name, _ in emissions
        )

    # Retain every component/principal/origin coordinate, including zero-gas reagent pools.
    for branch_name, fraction in (("flare", flare), ("chp", chp)):
        branch = gas.trace[branch_name]
        rows = {(x["component"], x["principal"], x["origin"]): x for x in branch["carbon"]}
        assert rows.keys() == pools.keys()
        for key, (methane_c, raw_co2_c) in pools.items():
            row = rows[key]
            assert row["methane_carbon_kg"] == pytest.approx(fraction * methane_c, rel=2e-12, abs=0)
            assert row["carbon_dioxide_carbon_kg"] == pytest.approx(
                fraction * raw_co2_c, rel=2e-12, abs=0
            )
            remaining = row["carbon_dioxide_output_carbon_kg"] - fraction * raw_co2_c
            exits = fsum(
                row[name]
                for name in (
                    "co_carbon_kg",
                    "methane_slip_carbon_kg",
                    "nmvoc_carbon_kg",
                    "particulate_carbon_kg",
                )
            )
            assert remaining + exits == pytest.approx(fraction * methane_c, rel=2e-12, abs=0)
            assert remaining >= 0
        products = branch["products_kg"]
        methane = fraction * fsum(x[0] for x in pools.values()) * 16 / 12
        ammonia = fraction * ammonia_n * 17 / 14
        h2s = fraction * sulfur * 34 / 32
        assert branch["input_species_kg"]["CH4"] == pytest.approx(methane, rel=2e-12, abs=0)
        # Balanced gas stoichiometry, Doka rounded molecular masses in kg/kmol;
        # anonymous NMVOC is CH1.85 and PM is soot-C. Water factor is 18/(2*1).
        incoming_h = methane / 4 + ammonia * 3 / 17 + h2s * 2 / 34
        outgoing_h = products["methane"] / 4 + products["nmvoc"] * 1.85 / 13.85
        outgoing_h += branch["plant_water_kg"] * 2 / 18
        assert outgoing_h == pytest.approx(incoming_h, rel=2e-12, abs=0)
        assert products["so2"] / 2 == pytest.approx(fraction * sulfur, rel=2e-12, abs=0)
        new_co2_c = fsum(
            row["carbon_dioxide_output_carbon_kg"] - row["carbon_dioxide_carbon_kg"]
            for row in rows.values()
        )
        # Raw inlet CO2 oxygen passes through; only newly oxidized carbon uses O2.
        oxygen = new_co2_c * 32 / 12 + products["co"] * 16 / 28
        oxygen += branch["plant_water_kg"] * 16 / 18 + products["so2"] / 2
        oxygen += products["nox"] * 32 / 46 + products["n2o"] * 16 / 44
        assert branch["plant_oxygen_kg"] == pytest.approx(oxygen, rel=2e-12, abs=0)
        oxide_n = products["nox"] * 14 / 46 + products["n2o"] * 28 / 44
        assert branch["gas_nitrogen_to_oxides_kg"] + branch[
            "gas_dinitrogen_out_kg"
        ] == pytest.approx(fraction * nitrogen, rel=2e-12, abs=0)
        assert branch["gas_nitrogen_to_oxides_kg"] + branch[
            "air_nitrogen_to_oxides_kg"
        ] == pytest.approx(oxide_n, rel=2e-12, abs=0)
        # Doka combustion air is 76/24 N2/O2 by mass. N-free landfill gas can produce thermal NOx
        # from air nitrogen.
        air_n = branch["additional_oxygen_kg"] * 0.76 / 0.24
        if stage == "landfill":
            air_n += fraction * gas.trace["source_equivalent_volume_m3"] * 0.114
        assert branch["air_dinitrogen_out_kg"] + branch[
            "air_nitrogen_to_oxides_kg"
        ] == pytest.approx(air_n, rel=2e-12, abs=0)
        assert products["nox"] > 0
        if branch_name == "chp" and date:
            assert products["n2o"] == 0  # explicit zero external factor, no division by zero

    for native_origin in ("fossil", "non-fossil"):
        selected = {key for key in pools if (key[2] == "fossil") == (native_origin == "fossil")}
        incoming_c = fsum(fsum(pools[key]) for key in selected)
        emitted_c = fsum(
            emissions.get((f"{species}, {native_origin}", part), 0) * c_fraction
            for species, c_fraction in (
                ("Carbon dioxide", 12 / 44),
                ("Carbon monoxide", 12 / 28),
                ("Methane", 12 / 16),
            )
            for part in (RURAL, URBAN)
        )
        # NMVOC/PM each use one native flow; their origin-specific carbon is retained in the ledger.
        emitted_c += fsum(
            row["nmvoc_carbon_kg"] + row["particulate_carbon_kg"]
            for name in ("flare", "chp")
            for row in gas.trace[name]["carbon"]
            if (row["component"], row["principal"], row["origin"]) in selected
        )
        assert emitted_c == pytest.approx(incoming_c, rel=2e-12, abs=0)
    # CO2/CO/CH4 identities combine biogenic and DAC emissions; principal-origin tags retain their
    # separate attribution. Protein and wastewater PAM contribute through auxiliary carbon.
    # NMVOC/soot retain their separate ledger.
    carbon_fractions = {
        f"{species}, {native_origin}": fraction
        for species, fraction in (
            ("Carbon dioxide", 12 / 44),
            ("Carbon monoxide", 12 / 28),
            ("Methane", 12 / 16),
        )
        for native_origin in ("fossil", "non-fossil")
    }
    carbon_ledgers = {
        branch: cast(list[dict[str, Any]], cast(dict[str, Any], gas.trace[branch])["carbon"])
        for branch in ("flare", "chp")
    }
    for index, carbon_origin in enumerate(("fossil", "biogenic", "atmospheric_dac")):
        incoming = fsum(
            fsum(masses)
            for (_, principal, source_origin), masses in pools.items()
            if principal and source_origin == carbon_origin
        )
        anonymous = fsum(
            row["nmvoc_carbon_kg"] + row["particulate_carbon_kg"]
            for branch in ("flare", "chp")
            for row in carbon_ledgers[branch]
            if row["principal"] and row["origin"] == carbon_origin
        )
        tagged = fsum(
            cast(float, row["amount"])
            * carbon_fractions[cast(str, row["name"])]
            * cast(tuple[float, float, float], row["principal_carbon_origin_fractions"])[index]
            for row in gas.biosphere_exchanges
            if row["name"] in carbon_fractions
        )
        assert tagged == pytest.approx(incoming - anonymous, rel=2e-12, abs=0)
    for name, key in (("Water", "plant_water_kg"), ("Oxygen", "plant_oxygen_kg")):
        total = fsum(gas.trace[branch][key] for branch in ("flare", "chp"))
        observed = fsum(value for (flow, _), value in emissions.items() if flow == name)
        assert observed == pytest.approx(total / (1000 if name == "Water" else 1), rel=2e-12, abs=0)
    assert fsum(
        value for (name, _), value in emissions.items() if name == "Sulfur dioxide"
    ) == pytest.approx(2 * sulfur, rel=2e-12, abs=0)
    assert fsum(
        value for (name, _), value in emissions.items() if name == "Ammonia"
    ) == pytest.approx(leak * ammonia_n * 17 / 14, rel=2e-12, abs=0)
    for name in {name for name, _ in emissions if name != "Oxygen"}:
        total = emissions.get((name, RURAL), 0) + emissions.get((name, URBAN), 0)
        assert emissions.get((name, URBAN), 0) == pytest.approx(urban * total, rel=2e-12, abs=0)
    assert all("properties" not in row for row in gas.biosphere_exchanges)
    assert not any(name == "Heat, waste" for name, _ in emissions)  # parent owns heat binding


@pytest.mark.parametrize("stage", ("landfill", "wastewater"))
def test_gas_normalization_energy_and_linked_lci(gas_boundary, gas_loads, stage):
    """Use dated factor/fuel ratios and allocate actual station energy once."""
    bd, context, database, date = gas_boundary
    loads = gas_loads[1]["fossil"]
    gas = _compile(context, stage, loads)
    pools, nitrogen, sulfur, ammonia_n, (leak, flare, chp) = _inputs(stage, loads)
    methane_c = fsum(x[0] for x in pools.values())
    raw_co2_c = fsum(x[1] for x in pools.values())
    if stage == "landfill":
        # Doka2003 Table 6.2 gas kg/m3; source-equivalent volume differs from
        # dry volume. Factor transfer uses 50 MJ/kg CH4 and 518.17 MJ/kmol H2S.
        volume = (methane_c / 12 + raw_co2_c / 12 + nitrogen / 28 + sulfur / 32) / (
            0.3357 / 16 + 0.7268 / 44 + 0.049 / 28
        )
        energy = methane_c * 16 / 12 * 50 + sulfur / 32 * 518.17
        engine_units = chp * energy / (CHP_FUEL[date] * CHP_LHV)
        furnace_units = flare * energy / (FURNACE_FUEL[date] * FURNACE_LHV)
        assert gas.trace["source_equivalent_volume_m3"] == pytest.approx(volume, rel=2e-12, abs=0)
    else:
        # Doka2021 gives dry normal m3/kmol. NH3 contains one N atom;
        # N2 uses a molar mass of 28 kg/kmol.
        volume = 22.4138 * (
            (methane_c + raw_co2_c) / 12
            + ammonia_n / 14
            + (nitrogen - ammonia_n) / 28
            + sulfur / 32
        )
        engine_units = chp * volume / CHP_FUEL[date]
        furnace_units = flare * volume * CHP_LHV / (FURNACE_FUEL[date] * FURNACE_LHV)
        assert gas.trace["raw_dry_normal_volume_m3"] == pytest.approx(volume, rel=2e-12, abs=0)
    assert gas.trace["chp"]["native_multiplier"] == pytest.approx(
        engine_units / CHP_PRODUCTION[date], rel=2e-12, abs=0
    )
    assert gas.trace["flare"]["native_multiplier"] == pytest.approx(
        furnace_units / FURNACE_PRODUCTION[date], rel=2e-12, abs=0
    )
    for name, species in (
        ("Carbon monoxide, non-fossil", "co"),
        ("Methane, non-fossil", "methane"),
        (NMVOC, "nmvoc"),
        ("Nitrogen oxides", "nox"),
        ("Dinitrogen monoxide", "n2o"),
        ("Platinum", "platinum"),
    ):
        assert gas.trace["chp"]["products_kg"][species] == pytest.approx(
            ENGINE_FACTORS[name][date] * engine_units, rel=2e-12, abs=0
        )
    assert gas.trace["flare"]["products_kg"]["n2o"] == pytest.approx(
        FURNACE_N2O[date] * furnace_units, rel=2e-12, abs=0
    )
    # Doka2003 Table 6.3 kg/m3 flare CO, total unburned hydrocarbons and NOx.
    flare_products = gas.trace["flare"]["products_kg"]
    assert flare_products["co"] == pytest.approx(0.00014 * flare * volume, rel=2e-12, abs=0)
    assert flare_products["methane"] + flare_products["nmvoc"] == pytest.approx(
        0.000014 * flare * volume, rel=2e-12, abs=0
    )
    assert flare_products["methane"] / flare_products["nmvoc"] == pytest.approx(0.0018 / 0.00042)
    assert flare_products["nox"] == pytest.approx(0.000419 * flare * volume, rel=2e-12, abs=0)
    # Landfill source 47 mg/m3; WWcalc EC14 retains the unrounded WWT mg/m3.
    particles = 0.000047 if stage == "landfill" else 46.74857142857142 / 1e6
    for branch, fraction in (("flare", flare), ("chp", chp)):
        assert gas.trace[branch]["products_kg"]["particles"] == pytest.approx(
            particles * fraction * volume, rel=2e-12, abs=0
        )

    if stage == "landfill":
        # Doka power retains source-equivalent volume and gas-C scaling.
        # The dry-normal volume includes entrained air separately.
        carbon_reference = 0.3357 * 12 / 16 + 0.7268 * 12 / 44
        pump = 0.53 * volume * carbon_reference * 0.046647169657919804
        generated = 0.278 * 0.3498 * energy / 3.6  # Doka gross electricity, kWh
        physical_volume = 22.4 * (
            methane_c / 12
            + raw_co2_c / 12
            + nitrogen / 28
            + sulfur / 32
            + volume * (0.114 / 28 + 0.036 / 32)
        )
        assert gas.trace["physical_dry_volume_m3"] == pytest.approx(
            physical_volume, rel=2e-12, abs=0
        )
        assert physical_volume != pytest.approx(volume)
        assert gas.trace["extraction_electricity_kwh"] == pytest.approx(pump, rel=2e-12, abs=0)
        assert pump < generated
        export = generated - pump
        assert gas.trace["net_electricity_export_kwh_no_credit"] == pytest.approx(
            export, rel=2e-12, abs=0
        )
    else:
        assert gas.heat_soil_mj == 0
        assert gas.trace["delivered_electricity_kwh"] == gas.trace["delivered_heat_mj"] == 0
        export = 0.0
    # Native transferred emission uncertainty follows the same positive ratio;
    # CO2/leakage/nominal chemistry remain deterministic, with no donor assay.
    transferred = [row for row in gas.biosphere_exchanges if row["uncertainty type"] == 2]
    assert transferred
    for row in transferred:
        assert row["loc"] == pytest.approx(log(row["amount"]) + LOG_OFFSET, rel=2e-12, abs=0)
        assert row["scale"] == LOG_SPREAD

    # Chase (1998), via NIST Chemistry WebBook 298.15 K liquid-water formation energies (MJ/kmol),
    # gives reaction HHV. Michel (via Doka) supplies the CH1.85 NMVOC allocation. Emission factors
    # use the fixture's separate gas LHV.
    hhv = {
        "CH4": (-74.87 + 393.52 + 2 * 285.83) / 16,
        "CO": (-110.53 + 393.52) / 28,
        "C": 393.52 / 12,
        "NH3": (-45.90 + 1.5 * 285.83) / 17,
        "H2S": (-20.50 + 296.84 + 285.83) / 34,
        "NMVOC": (12 * 34.016 + 1.85 * 124.265) / 13.85,
    }
    reaction = 0.0
    for branch in (gas.trace["flare"], gas.trace["chp"]):
        products = branch["products_kg"]
        fuel = fsum(
            hhv[species] * branch["input_species_kg"][species] for species in ("CH4", "NH3", "H2S")
        )
        residual = fsum(
            hhv[species] * products[key]
            for species, key in (
                ("CH4", "methane"),
                ("CO", "co"),
                ("C", "particles"),
                ("NMVOC", "nmvoc"),
            )
        )
        oxide = products["nox"] * 33.10 / 46 + products["n2o"] * 82.05 / 44
        assert branch["reaction_hhv_mj"] == pytest.approx(fuel - residual - oxide, rel=2e-12, abs=0)
        reaction += fuel - residual - oxide
    assert gas.heat_air_mj == pytest.approx(reaction - export * 3.6, rel=2e-12, abs=0)
    if stage == "landfill":
        residue = loads[0]
        # Michel (via Doka2021 WWcalc DT18), MJ/kg of organic element only at this
        # landfill decomposition boundary. Unchanged soluble reagents cancel.
        organic = [
            x for x in residue.components if x.component in {"glucan", "glycerol", "protein"}
        ]
        coefficients = {"C": 34.016, "H": 124.265, "O": -9.8324, "N": 6.276, "S": 19.079}
        acetate_carbon = fsum(
            x.leachate_carbon_kg
            for x in residue.carbon
            if x.component in {"glucan", "glycerol", "protein"}
        )
        # The admitted dissolved organic reference is acetic acid, C2H4O2;
        # its H/C and O/C mass ratios use the source's atomic masses.
        acetate_elements = {
            "C": acetate_carbon,
            "H": acetate_carbon * 4 * 1.008 / (2 * 12.01),
            "O": acetate_carbon * 2 * 16 / (2 * 12.01),
        }
        remaining_energy = fsum(
            coefficient
            * (
                fsum(x.elements_kg.get(element, 0) for x in organic)
                - acetate_elements.get(element, 0)
            )
            for element, coefficient in coefficients.items()
        )
        remaining_energy -= 6.276 * (fsum(x.elements_kg.get("N", 0) for x in organic) - nitrogen)
        remaining_energy -= 19.079 * (fsum(x.elements_kg.get("S", 0) for x in organic) - sulfur)
        remaining_energy -= methane_c * 16 / 12 * hhv["CH4"] + sulfur * 34 / 32 * hhv["H2S"]
        assert gas.heat_soil_mj == pytest.approx(remaining_energy, rel=2e-12, abs=0)

    # A one-kg gas contribution node links inventories on its parent's reference basis.
    graph_name = "gas_" + stage
    key = graph_name, "actual_gas"
    bd.Database(graph_name).write(
        {
            key: {
                "name": "Actual TPS " + stage + " gas contribution",
                "reference product": "gas contribution",
                "unit": "kilogram",
                "location": "RoW",
                "type": "process",
                "exchanges": [
                    exchange(key, 1.0, "production"),
                    *gas.biosphere_exchanges,
                ],
            }
        },
        searchable=False,
    )
    lca = LCA({bd.get_node(database=graph_name, code="actual_gas"): 1})
    lca.lci()
    for code in ("chp", "furnace", "biogas", "natural_gas", "electricity"):
        node = bd.get_node(database=database, code=code)
        assert node.id not in lca.dicts.activity
    direct_carbon_dioxide = fsum(
        value
        for (name, _), value in _emissions(bd, gas).items()
        if name == "Carbon dioxide, fossil"
    )
    total_carbon_dioxide = fsum(
        float(lca.inventory[lca.dicts.biosphere[flow.id], :].sum())
        for flow in bd.Database("biosphere")
        if flow["name"] == "Carbon dioxide, fossil" and flow.id in lca.dicts.biosphere
    )
    assert total_carbon_dioxide == pytest.approx(direct_carbon_dioxide, rel=3e-7, abs=0)
