"""Detached PA gas preserves the complete real native binding boundary.

Invented, fully linked gas sources reuse the existing gas fixture. Original and
future graphs have different nonunit production, uncertainty and a genuine zero
N2O template. Every future activity code is reassigned before native admission.
Actual PA6/PA66 plus retained finish supplies all three principal carbon origins.
The source projects are closed before the detached owner is evaluated.
"""

import json
from dataclasses import asdict
from typing import cast

import pytest
from material_recipe_fixtures import physical_material_recipe

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
    _CH_WASTEWATER_PROFILE,
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_load import (
    _pa_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingGasReference,
    _admit_gas_sources,
    _gas_fuel_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.waste_gas import (
    _landfill_gas,
    _wastewater_gas,
)
from pytoce.run_lca.calculation.pa_gas_receiving import (
    receive_pa_landfill_gas,
    receive_pa_wastewater_gas,
)
from tests.unit.prepare_backgrounds.tps_gas_fixtures import (
    GAS_SOURCES,
    biosphere_inventory,
    native_inventory,
)


@pytest.fixture(scope="module", params=(False, True), ids=("original-pa6", "rekeyed-future-pa66"))
def admitted_gas(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    """Admit real source records and retain live-owner observations before closing the store."""
    future = bool(request.param)
    root = tmp_path_factory.mktemp("pa_gas_receiving")
    catalogue = load_provider_catalogue()
    construction: dict[str, object] = {"gas_sources": GAS_SOURCES}
    # The shared material calculation supplies the retained seven-ingredient finish.
    finish = _pa_finish_reference("pa66" if future else "pa6")
    mixture = _pa_landfill_load(
        finish,
        principal_by_origin_kg={"fossil": 0.5, "biogenic": 0.75, "atmospheric_dac": 0.75},
        finish_dose_kg=5.0,
        water_kg=0.4,
        site=_LandfillSite(16.743579766537, 1250.84108527132, 1.0),
    )
    # Synthetic hydraulics use the mixture reference. Both process profiles exercise their gas/air
    # allocation.
    waters = {
        region: _wastewater_load(
            mixture.leachate,
            volume_m3=0.002,
            incoming_water_kg=1.8,
            evaporation_fraction=0.01,
            profile=profile,
        )
        for region, profile in (("CH", _CH_WASTEWATER_PROFILE), ("RoW", _ROW_WASTEWATER_PROFILE))
    }
    names = BrightwayNames(root / "brightway", "pa_gas_receiving", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        bd.Database("original").write(native_inventory("original", future=False), searchable=False)
        database = "future" if future else "original"
        if future:
            native = native_inventory(database, future=True)
            keys = {key: (key[0], "dated-" + key[1]) for key in native}
            # Reassign the actual source graph, including every production and
            # internal input. Metadata resolution must retain these new keys.
            rekeyed = {}
            for key, record in native.items():
                record["code"] = keys[key][1]
                for exchange in record["exchanges"]:
                    exchange["input"] = keys.get(exchange["input"], exchange["input"])
                rekeyed[keys[key]] = record
            bd.Database(database).write(rekeyed, searchable=False)
        original = _ActivityResolver(bd, "original")
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="providers",
            source=original,
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
        reference = PaReceivingGasReference(
            _gas_fuel_reference(context, dict(GAS_SOURCES["chp"]), "biogas"),
            _gas_fuel_reference(
                context, dict(GAS_SOURCES["furnace"]), "natural gas, high pressure"
            ),
        )
        sources = _admit_gas_sources(context, construction)
        observed = {"landfill": _landfill_gas(context, construction, mixture.gas, "pa_gas_test")}
        observed.update(
            {
                region: _wastewater_gas(
                    context, construction, waters[region], "pa_gas_test", profile
                )
                for region, profile in (
                    ("CH", _CH_WASTEWATER_PROFILE),
                    ("RoW", _ROW_WASTEWATER_PROFILE),
                )
            }
        )
    # Nothing returned here contains a Brightway object. All receiving below
    # runs after project selection exits, using the actual admitted records.
    return future, mixture, waters, reference, sources, observed


@pytest.mark.parametrize("stage", ("landfill", "CH", "RoW"))
def test_detached_pa_gas_matches_complete_native_binding_and_origin_pools(admitted_gas, stage):
    """Preserve full quantities, distributions, origin attribution and parent energy duties."""
    future, mixture, waters, reference, sources, observed = admitted_gas
    if stage == "landfill":
        result = receive_pa_landfill_gas(mixture.gas, reference, sources)
    else:
        profile = _CH_WASTEWATER_PROFILE if stage == "CH" else _ROW_WASTEWATER_PROFILE
        result = receive_pa_wastewater_gas(waters[stage], profile, reference, sources)
    live = observed[stage]
    prefix = "pa_landfill_gas" if stage == "landfill" else "pa_wastewater_gas"
    functions = {f"{prefix}_leakage", f"{prefix}_flare", f"{prefix}_chp"}
    assert {row.function for row in result.amounts} == functions
    # Leakage, flare, and CHP retain distinct contribution destinations. Leakage records its
    # calibration-source provenance.
    for row in result.amounts:
        assert isinstance(row.endpoint, PaReceivingFlow)
        if row.function.endswith("_leakage"):
            assert row.source == reference.chp.activity
        elif row.function.endswith("_flare"):
            assert row.source == sources.furnace.activity
        elif row.function.endswith("_chp"):
            assert row.source == sources.chp.activity
    for field in (
        "heat_air_mj",
        "heat_soil_mj",
        "internal_electricity_kwh",
        "internal_heat_mj",
    ):
        assert getattr(result, field) == pytest.approx(getattr(live, field), rel=2e-12, abs=0)
    rows = [
        {
            **row.metadata,
            "input": row.endpoint.key,
            "amount": row.amount,
            "principal_carbon_origin_fractions": row.principal_carbon_origin_fractions,
        }
        for row in result.amounts
    ]
    native_rows = live.biosphere_exchanges

    def coordinate(row):
        """Order actual origin-resolved outputs independently of source-row ordering."""
        return (
            row["input"],
            row.get("principal_carbon_origin_fractions") or (-1, -1, -1),
            row["amount"],
        )

    actual = sorted(rows, key=coordinate)
    expected = sorted(native_rows, key=coordinate)
    assert len(actual) == len(expected)
    for row, native in zip(actual, expected, strict=True):
        assert row["input"] == native["input"]
        assert row["amount"] == pytest.approx(native["amount"], rel=2e-12, abs=0)
        assert row.get("principal_carbon_origin_fractions") == native.get(
            "principal_carbon_origin_fractions"
        )
        for field in (
            "uncertainty type",
            "negative",
            "scale",
            "shape",
            "minimum",
            "maximum",
            "loc",
        ):
            assert (field in row) == (field in native)
            if field in row:
                assert row[field] == pytest.approx(native[field], rel=2e-12, abs=1e-15)
    assert json.loads(json.dumps(asdict(result), allow_nan=False))
    # Doka leakage retains its original calibration source. Treatment source
    # keys follow the selected background year.
    for row in result.amounts:
        if row.function.endswith("_leakage"):
            assert row.source == reference.chp.activity
        else:
            assert row.source in (sources.chp.activity, sources.furnace.activity)
            assert row.source.key[0] == ("future" if future else "original")
            assert row.source.key[1].startswith("dated-") == future
    attributed = [
        row for row in result.amounts if row.principal_carbon_origin_fractions is not None
    ]
    assert {row.principal_carbon_origin_fractions for row in attributed} == {
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
    }
    for row in attributed:
        assert isinstance(row.endpoint, PaReceivingFlow)
        fractions = cast(tuple[float, float, float], row.principal_carbon_origin_fractions)
        if fractions[0]:
            assert row.endpoint.name.endswith(", fossil")
        elif fractions[1] or fractions[2]:
            assert row.endpoint.name.endswith(", non-fossil")
    # Scaling preserves zero coefficients and species-specific attribution on the admitted
    # normalization.
    scaled = tuple(row.scaled(0.37) for row in result.amounts)
    for received, original in zip(scaled, result.amounts, strict=True):
        assert received.function == original.function
        assert received.source == original.source
        assert received.endpoint == original.endpoint
        assert received.amount == 0.37 * original.amount
        assert received.metadata is not original.metadata
        assert (
            received.principal_carbon_origin_fractions == original.principal_carbon_origin_fractions
        )
