"""Keep original PA receiving references and dated operating inputs portable."""

import json
from math import log
from pathlib import Path
from typing import Any, cast

import pytest
from pa_event_receiving_fixtures import pa_event_receiving_boundary
from tps_gas_fixtures import (
    CHP_FUEL,
    CHP_LHV,
    CHP_PRODUCTION,
    ENGINE_FACTORS,
    FURNACE_FUEL,
    FURNACE_LHV,
    FURNACE_PRODUCTION,
    GAS_SOURCES,
    LOG_OFFSET,
    LOG_SPREAD,
)
from tps_gas_fixtures import (
    biosphere_inventory as gas_biosphere_inventory,
)

from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    reference_from_payload,
    reference_to_payload,
    sources_from_payload,
    sources_to_payload,
)
from pytoce.prepare_backgrounds.preparation.providers.tps_residue_load import (
    _tps_landfill_gas_load,
    _tps_residue_load,
)
from pytoce.prepare_backgrounds.preparation.providers.waste_gas import (
    _landfill_gas,
    _wastewater_gas,
)
from pytoce.prepare_backgrounds.preparation.providers.waste_gas_load import (
    _landfill_gas_reference,
    _wastewater_gas_reference,
)


@pytest.mark.parametrize("original", [True, False])
def test_prepared_thermal_sources_retain_calibration_and_detached_dated_inputs(
    tmp_path: Path, original: bool
) -> None:
    """Admit original/future fixture inventories and reuse JSON outside the project."""
    with pa_event_receiving_boundary(tmp_path, original=original) as boundary:
        context = boundary.context
        reference, sources = boundary.reference, boundary.sources
        services = boundary.services
        gas_keys = boundary.gas_keys
        database = context.ordinary.database_name
        assert reference.hwi.key == ("original", "mc_hwi")
        assert reference.ash.key == ("original", "mc_ash")
        assert {
            element: reference.donor_elements_kg[element] for element in ("C", "H", "O", "N", "S")
        } == {
            "C": 0.3,
            "H": 0.06,
            "O": 0.04,
            "N": 0.01,
            "S": 0.02,
        }
        assert reference.ash_dry_fraction == 1
        assert reference.ash_element_contents == {
            "carbon": 0.05,
            "nitrogen": 0.03,
            "phosphorus": 0.025,
            "sulfur": 0.012,
            "sodium": 0.1,
            "calcium": 0.2,
            "vanadium": 0.02,
            "iron": 0.04,
            "nickel": 0.02,
            "silicon": 0.03,
            "aluminium": 0.07,
            "potassium": 0.11,
            "magnesium": 0.035,
        }
        assert sources.background_scenario == "remind_ssp2_npi"
        assert reference.water.wastewater.key == ("original", "mibk_plant_ch")
        assert reference.water.sludge.operation.activity.key == ("original", "organic_ch_sludge")
        assert sources.water.wastewater.activity.key == (database, "mibk_plant_ch")
        assert sources.water.wastewater.production_amount == (-7 if original else -14)
        assert reference.landfill.landfill.key == ("original", "landfill_source_site")
        assert reference.landfill.leachate_volume_m3_per_wet_kg == 0.006
        assert reference.landfill.wastewater.key == ("original", "landfill_source_plant")
        assert reference.landfill.sludge.operation.activity.key == (
            "original",
            "landfill_source_thermal_sludge",
        )
        assert "P" not in reference.landfill.sludge.elements_kg
        assert sources.landfill.landfill.activity.key == (database, "landfill_source_site")
        assert sources.landfill.landfill.production_amount == (-4 if original else -9)
        assert sources.landfill.wastewater.activity.key == (database, "landfill_source_plant")
        assert sources.landfill.wastewater.production_amount == (-3 if original else -7)
        assert sources.landfill.pam == sources.pam
        assert reference.recycling.preparation.key == (
            "original",
            "recycling_source_recycling_weee_preparation",
        )
        assert reference.recycling.dry_feed_kg_per_reference_kg == 1.6
        assert reference.recycling.wash_water_kg_per_reference_kg == pytest.approx(4.8)
        assert sources.recycling.preparation.activity.key == (
            "frontier",
            provider_service_code("recycling_weee_preparation"),
        )
        assert sources.recycling.finishing.activity.key == (
            "frontier",
            provider_service_code("recycling_pet_finishing"),
        )
        assert sources.recycling.preparation.production_amount == (2 if original else 5)
        assert sources.recycling.pellet_delivery.key == (
            "frontier",
            provider_service_code("recycling_pet_pellet_delivery"),
        )
        assert reference.municipal.incineration.activity.key == (
            "original",
            "municipal_source_phb_furnace",
        )
        assert reference.municipal.incineration.production_amount == -2
        assert sources.municipal.incineration.activity.key == (
            database,
            "municipal_source_phb_furnace",
        )
        assert sources.municipal.incineration.production_amount == (-2 if original else -5)
        assert sources.municipal.bottom_ash_treatment.production_amount == (-3 if original else -7)
        assert sources.municipal.pam == sources.pam
        assert {
            product: amount
            for product, amount in sources.municipal.bottom_ash_market.supplies().items()
            if product.startswith("transport,")
        } == pytest.approx(
            {
                "transport, freight, lorry": 0.02,
                "transport, freight train": 0.03,
                "transport, freight, inland waterways, barge": 0.027,
            }
        )
        # Unmanaged sources are admitted only from original data. The future
        # fixture deliberately contains no duplicate dump or burning nodes.
        assert reference.open_burning.activity.key == ("original", "burning_source_burning")
        assert [site.placement.activity.key for site in reference.dump.sites] == [
            ("original", f"dump_source_site_{index}") for index in range(5)
        ]
        assert [site.climate.mean_annual_precipitation_mm for site in reference.dump.sites] == [
            350,
            550,
            900,
            1100,
            1900,
        ]
        assert sources.provider_anchor_year == (2025 if original else 2050)
        assert sources.general_background_database == database
        assert sources.general_background_anchor_year == (2025 if original else 2050)
        assert sources.hwi.activity == PaReceivingActivity(
            key=(database, "mc_hwi"),
            name="treatment of hazardous waste, hazardous waste incineration",
            reference_product="hazardous waste, for incineration",
            unit="kilogram",
            location="CH",
        )
        assert sources.fixed.activity.key == (database, "mc_fixed")
        assert sources.ash.activity.key == (database, "mc_ash")
        assert (
            sources.hwi.production_amount,
            sources.fixed.production_amount,
            sources.ash.production_amount,
        ) == ((-3, 4, -5) if original else (-6, 8, -10))
        peroxide = next(
            row for row in sources.hwi.rows if row.endpoint.key == (database, "cr_peroxide")
        )
        assert peroxide.endpoint == PaReceivingActivity(
            key=(database, "cr_peroxide"),
            name="invented cr_peroxide supply",
            reference_product="hydrogen peroxide, without water, in 50% solution state",
            unit="kilogram",
            location="GLO",
        )
        peroxide_amount = 0.004 if original else 0.006
        assert peroxide.amount == pytest.approx(peroxide_amount)
        assert peroxide.metadata["allocation"] == "invented chemical source"
        assert peroxide.metadata["uncertainty type"] == 2
        assert peroxide.metadata["loc"] == pytest.approx(log(peroxide_amount) + 0.02)
        assert peroxide.metadata["scale"] == 0.17
        assert peroxide.metadata["minimum"] == pytest.approx(peroxide_amount / 3)
        assert peroxide.metadata["maximum"] == pytest.approx(peroxide_amount * 3)
        if not original:
            ammonia_row = next(
                row for row in sources.hwi.rows if row.endpoint.key == (database, "mc_ammonia")
            )
            assert ammonia_row.amount == pytest.approx(0.0001)
            assert ammonia_row.metadata["loc"] == pytest.approx(0.0001)
            assert ammonia_row.metadata["uncertainty type"] == 0
            assert ammonia_row.metadata["negative"] is False
            assert not {"scale", "shape", "minimum", "maximum"}.intersection(ammonia_row.metadata)
        source_co2 = next(
            row for row in sources.hwi.rows if row.endpoint.key == ("biosphere", "mc_co2_fossil")
        )
        assert source_co2.endpoint == PaReceivingFlow(
            key=("biosphere", "mc_co2_fossil"),
            name="Carbon dioxide, fossil",
            unit="kilogram",
            categories=("air", "urban air close to ground"),
        )
        assert source_co2.amount == 1
        added = {(flow.name, flow.categories): flow for flow in sources.added_biosphere}
        for code, name, unit, categories in (
            (
                "pa_spinning_nmvoc",
                "NMVOC, non-methane volatile organic compounds",
                "kilogram",
                ("air",),
            ),
            (
                "mc_co2_bio",
                "Carbon dioxide, non-fossil",
                "kilogram",
                ("air", "urban air close to ground"),
            ),
            (
                "mc_co_bio",
                "Carbon monoxide, non-fossil",
                "kilogram",
                ("air", "urban air close to ground"),
            ),
            ("cr_nitrogen_air", "Nitrogen", "kilogram", ("air",)),
            ("mc_heat_water", "Heat, waste", "megajoule", ("water", "surface water")),
        ):
            assert added[name, categories] == PaReceivingFlow(
                key=("biosphere", code), name=name, unit=unit, categories=categories
            )
        assert sources.reheat_product == (
            "heat, district or industrial, natural gas"
            if original
            else "heat, district or industrial"
        )
        assert sources.pam == PaReceivingActivity(
            key=(database, "mc_pam"),
            name="market for polyacrylamide",
            reference_product="polyacrylamide",
            unit="kilogram",
            location="GLO",
        )
        for operation in (sources.hwi, sources.fixed):
            assert {"titanium dioxide", "chromium"} <= operation.supplies().keys()
            assert {
                row.endpoint.key[0]
                for row in operation.rows
                if isinstance(row.endpoint, PaReceivingActivity)
                and row.endpoint.reference_product in {"titanium dioxide", "chromium"}
            } == {database}
        for operation in (
            sources.water.sludge_incineration,
            sources.landfill.sludge_incineration,
            sources.municipal.incineration,
        ):
            assert {"titanium dioxide", "chromium trioxide, flakes"} <= operation.supplies().keys()
        date = int(not original)
        for role, fuel, lhv, native_size, fuel_amount in (
            ("chp", "biogas", CHP_LHV, CHP_PRODUCTION[date], CHP_FUEL[date]),
            ("furnace", "natural_gas", FURNACE_LHV, FURNACE_PRODUCTION[date], FURNACE_FUEL[date]),
        ):
            calibration = getattr(reference.gas, role)
            operation = getattr(sources.gas, role)
            assert calibration.activity.key == ("original", role)
            assert calibration.fuel.key == ("original", fuel)
            assert calibration.fuel.unit == "cubic meter"
            assert calibration.net_heat_mj_per_fuel_unit == lhv
            assert operation.activity.key == (database, role)
            assert operation.activity.unit == "megajoule"
            assert operation.production_amount == native_size
            fuel_row = next(row for row in operation.rows if row.endpoint.key == (database, fuel))
            assert fuel_row.amount == pytest.approx(fuel_amount)
            assert fuel_row.metadata["loc"] == pytest.approx(log(fuel_amount) + LOG_OFFSET)
            assert fuel_row.metadata["scale"] == LOG_SPREAD
        # The gas record admits combustion calibration; donor plant support
        # stays outside it.
        assert sources.gas.chp.supplies().keys() == {"biogas"}
        assert sources.gas.furnace.supplies().keys() == {"natural gas, high pressure"}
        for name, coefficients in ENGINE_FACTORS.items():
            row = next(row for row in sources.gas.chp.rows if row.endpoint.name == name)
            assert row.amount == pytest.approx(coefficients[date])
        # Gas emissions share the chemical fixture's existing native identities;
        # their complete rural/urban set excludes parent-owned waste heat.
        expected_flows = {
            (row["name"], row["unit"], tuple(row["categories"])): gas_keys[key]
            for key, row in gas_biosphere_inventory().items()
            if row["name"] != "Heat, waste"
        }
        assert {
            (flow.name, flow.unit, flow.categories): flow.key for flow in sources.gas.biosphere
        } == expected_flows
        for operation in (
            sources.hwi,
            sources.fixed,
            sources.ash,
            sources.gas.chp,
            sources.gas.furnace,
        ):
            for row in operation.rows:
                # Saved input identity uses the portable database/code pair.
                assert not {"id", "input", "output"}.intersection(row.metadata)
        # TPS loads supply mixed-origin gas for admission and detached calculation checks.
        # PA mixture chemistry is exercised by its receiving tests.
        residue = _tps_residue_load(services, "tps_fossil")
        landfill_load = _tps_landfill_gas_load(residue)
        water_load = _wastewater_load(
            residue.leachate,
            volume_m3=0.002,
            incoming_water_kg=1.8,
            evaporation_fraction=0.01,
            profile=_CH_WASTEWATER_PROFILE,
        )
        gas_construction: dict[str, object] = {"gas_sources": GAS_SOURCES}
        live_landfill = _landfill_gas(context, gas_construction, landfill_load, "pa_gas_admission")
        live_water = _wastewater_gas(
            context, gas_construction, water_load, "pa_gas_admission", _CH_WASTEWATER_PROFILE
        )
        reference_payload = reference_to_payload(reference)
        sources_payload = sources_to_payload(sources)
        assert reference_payload["schema"] == "pytoce_pa_receiving_reference_schema_v1"
        assert sources_payload["schema"] == "pytoce_pa_receiving_sources_schema_v1"
        reference_json = json.dumps(reference_payload, allow_nan=False)
        sources_json = json.dumps(sources_payload, allow_nan=False)

    decoded_reference = reference_from_payload(json.loads(reference_json))
    decoded_sources = sources_from_payload(json.loads(sources_json))
    assert decoded_reference == reference
    assert decoded_sources == sources
    assert decoded_reference.landfill.landfill.key[0] == "original"
    assert decoded_sources.landfill.wastewater.activity.key[0] == database
    assert decoded_sources.landfill.collection.supplies() == sources.landfill.collection.supplies()
    assert decoded_reference.municipal.incineration.activity.key[0] == "original"
    assert decoded_sources.municipal.incineration.activity.key[0] == database
    assert decoded_reference.open_burning == reference.open_burning
    assert decoded_reference.dump == reference.dump
    thermal = decoded_sources.thermal_source(decoded_reference)
    assert thermal == sources.thermal_source(reference)
    assert thermal.donor_elements_kg == reference.donor_elements_kg
    assert thermal.waste_supplies[
        "hydrogen peroxide, without water, in 50% solution state"
    ] == pytest.approx(peroxide_amount)
    assert thermal.waste_supplies["heat, district or industrial, natural gas"] == pytest.approx(
        0.02
    )
    assert (thermal.oil_kg_per_mj, thermal.boiler_electricity_kwh_per_mj) == (0, 0)
    assert decoded_sources.ash_source(decoded_reference) == sources.ash_source(reference)
    chp = decoded_sources.gas.chp_operation(decoded_reference.gas)
    furnace = decoded_sources.gas.furnace_operation(decoded_reference.gas)
    landfill = _landfill_gas_reference(landfill_load, chp, furnace)
    wastewater = _wastewater_gas_reference(water_load, chp, furnace, _CH_WASTEWATER_PROFILE)
    assert decoded_sources.gas.flow_index == {
        (flow.name, flow.categories): flow for flow in decoded_sources.gas.biosphere
    }
    # The persisted scientific contract retains its declared input identity during lazy preparation.
    assert json.dumps(sources_to_payload(decoded_sources), allow_nan=False) == sources_json
    assert sources_from_payload(sources_to_payload(decoded_sources)) == decoded_sources
    assert landfill.volume_m3 == pytest.approx(live_landfill.trace["source_equivalent_volume_m3"])
    assert landfill.heat_air_mj == pytest.approx(live_landfill.heat_air_mj)
    assert wastewater.volume_m3 == pytest.approx(live_water.trace["raw_dry_normal_volume_m3"])
    assert wastewater.electricity_kwh == pytest.approx(live_water.internal_electricity_kwh)
    assert wastewater.useful_heat_mj == pytest.approx(live_water.internal_heat_mj)
    assert wastewater.heat_air_mj == pytest.approx(live_water.heat_air_mj)
    assert wastewater.electricity_kwh > 0
    assert wastewater.useful_heat_mj > 0
    for detached, live in ((landfill, live_landfill), (wastewater, live_water)):
        for branch, role in zip(detached.branches, ("flare", "chp"), strict=True):
            native = cast(dict[str, Any], live.trace[role])
            assert branch.products == pytest.approx(native["products_kg"], rel=2e-12, abs=0)
            assert branch.water_kg == pytest.approx(native["plant_water_kg"], rel=2e-12, abs=0)
            assert branch.oxygen_kg == pytest.approx(native["plant_oxygen_kg"], rel=2e-12, abs=0)
