"""Verify actual PA municipal receiving with complete invented native graphs."""

import json
import math
from dataclasses import asdict
from math import fsum
from typing import Any, cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from pa_municipal_fixtures import (
    SURFACE,
    URBAN,
    pa_municipal_biosphere,
    pa_municipal_native_inventory,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    pa_municipal_receiving_sources as municipal_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    load_provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.run_lca.calculation.pa_municipal_receiving import (
    receive_pa_municipal_incineration,
)
from pytoce.run_lca.calculation.pa_receiving import PaDisposalEvent


@pytest.fixture(params=[True, False], ids=["original", "future"])
def boundary(request: pytest.FixtureRequest, tmp_path):
    """Admit complete original/future functions, then expose detached records."""
    original = request.param
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    names = BrightwayNames(tmp_path / "brightway", "pa_municipal", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(pa_municipal_biosphere(), searchable=False)
        bd.Database("original").write(
            pa_municipal_native_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                pa_municipal_native_inventory(catalogue, database, original=False), searchable=False
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
        reference = municipal_sources.admit_pa_municipal_receiving_reference(context)
        sources = municipal_sources.admit_pa_municipal_receiving_sources(context, reference)
        reference_payload = json.loads(json.dumps(asdict(reference), allow_nan=False))
        sources_payload = json.loads(json.dumps(asdict(sources), allow_nan=False))
    restored = (
        municipal_sources.municipal_reference_from_payload(reference_payload),
        municipal_sources.municipal_sources_from_payload(sources_payload),
    )
    assert restored == (reference, sources)
    return (*restored, database, original)


def test_actual_pa_municipal_full_receiving(boundary):
    """Close both grades/origins and wet EOL versus already washed dry rejects."""
    reference, sources, database, original = boundary
    assert reference.incineration.production_amount == -2
    assert sources.incineration.production_amount == (-2 if original else -5)
    assert "P" not in reference.elements_kg
    for constituent in ("polyamide_6_resin", "polyamide_66_resin"):
        for dose, water, demand in (
            (
                3.2,
                (0.044 if constituent == "polyamide_6_resin" else 0.043) * 2.5,
                "incineration_treatment",
            ),
            (0.8, 0.0, "mechanical_recycling_reject_incineration_treatment"),
        ):
            event = PaDisposalEvent(2035, "WEU", constituent, demand, (1.2, 0.8, 0.5), dose, water)
            result = receive_pa_municipal_incineration(event, reference, sources)
            load = result.incineration
            assert load.material.water_kg == water
            assert fsum(event.principal_by_origin_kg) == 2.5
            assert load.drivers["process"] > 0
            assert load.thermal["process_water_kg"] > 0
            assert load.thermal["reaction_water_kg"] > 0
            assert load.thermal["environmental_heat_MJ"] > 0
            assert load.thermal["net_electricity_MJ"] > 0
            direct = [x for x in result.amounts if x.function == "pa_municipal_incineration"]
            purchases = [x for x in direct if isinstance(x.endpoint, PaReceivingActivity)]
            water_supply = [
                x
                for x in purchases
                if cast(PaReceivingActivity, x.endpoint).reference_product == "water, decarbonised"
            ]
            assert fsum(x.amount for x in water_supply) == pytest.approx(load.duties["water"])
            assert {x.endpoint.key[0] for x in water_supply} == {database}
            by_code = {x.endpoint.key[1]: x for x in water_supply}
            assert by_code["phb_water_a"].amount == pytest.approx(
                load.duties["water"] * (0.4 if original else 0.8)
            )
            assert by_code["phb_water_a"].metadata["loc"] == pytest.approx(
                math.log(by_code["phb_water_a"].amount) + 0.03
            )
            for product, duty in (
                ("titanium dioxide", "TiO2"),
                ("chromium trioxide, flakes", "V2O5"),
            ):
                catalyst = [
                    x
                    for x in purchases
                    if cast(PaReceivingActivity, x.endpoint).reference_product == product
                ]
                assert fsum(x.amount for x in catalyst) == pytest.approx(load.duties[duty])
                assert {x.endpoint.key[0] for x in catalyst} == {database}
            assert fsum(x.amount for x in purchases if x.endpoint == sources.pam) == pytest.approx(
                load.duties["PAM"]
            )
            assert not any(
                cast(PaReceivingActivity, x.endpoint).reference_product == "chemical, inorganic"
                for x in purchases
            )
            emissions = {}
            for x in direct:
                if isinstance(x.endpoint, PaReceivingFlow):
                    key = x.endpoint.name, x.endpoint.categories
                    emissions[key] = emissions.get(key, 0) + x.amount
            assert 1000 * fsum(
                emissions["Water", c] for c in (URBAN, SURFACE)
            ) + load.bottom_ash.water_kg == pytest.approx(
                water + load.thermal["reaction_water_kg"] + load.duties["water"]
            )
            assert emissions[
                "Non-hazardous waste disposed", ("inventory indicator", "waste")
            ] == pytest.approx(load.drivers["wet"])
            assert fsum(emissions["Heat, waste", c] for c in (URBAN, SURFACE)) == pytest.approx(
                load.thermal["environmental_heat_MJ"]
            )
            # Waste-N cyanide debits C once alongside every municipal trace;
            # PM/dioxin indicators overlap and supply no molecular carbon assay.
            c_air = fsum(
                emissions[f"{species}, {origin}", URBAN] * 12 / molecular
                for species, molecular in (
                    ("Carbon dioxide", 44),
                    ("Carbon monoxide", 28),
                    ("Methane", 16),
                )
                for origin in ("fossil", "non-fossil")
            )
            c_air += emissions["NMVOC, non-methane volatile organic compounds", URBAN] * 12 / 13.85
            c_air += emissions["Cyanide", URBAN] * 12 / 26
            for name, c, h, o, cl in (
                ("Benzene", 6, 6, 0, 0),
                ("Benzo(a)pyrene", 20, 12, 0, 0),
                ("Hexachlorobenzene", 6, 0, 0, 6),
                ("Pentachlorobenzene", 6, 1, 0, 5),
                ("Pentachlorophenol", 6, 1, 1, 5),
                ("Toluene", 7, 8, 0, 0),
            ):
                c_air += emissions[name, URBAN] * 12 * c / (12 * c + h + 16 * o + 35.45 * cl)
            retained = _pa_finish_reference(
                "pa6" if constituent == "polyamide_6_resin" else "pa66",
            ).retained
            c_in = 2.5 * 72 / 113 + dose * retained.elements_kg["C"]
            c_ash = fsum(
                pool.carbon_kg
                for ash in (load.bottom_ash, load.residual_ash)
                for pool in ash.carbon
                if pool.component != "APC_PAM"
            )
            assert c_air + emissions[
                "DOC, Dissolved Organic Carbon", SURFACE
            ] + c_ash == pytest.approx(c_in, rel=2e-14)
            sodium_rows = [
                x
                for x in direct
                if isinstance(x.endpoint, PaReceivingFlow)
                and x.endpoint.name == "Sodium I"
                and x.endpoint.categories == SURFACE
            ]
            assert sorted(cast(int, x.metadata["uncertainty type"]) for x in sodium_rows) == [0, 2]
            fresh_na = load.duties["NaOH"] * 22.99 / (22.99 + 15.999 + 1.008)
            assert next(
                x.amount for x in sodium_rows if x.metadata["uncertainty type"] == 0
            ) == pytest.approx(fresh_na)
            for species, molar in (
                ("Carbon dioxide", 44),
                ("Carbon monoxide", 28),
                ("Methane", 16),
            ):
                attributed = [0.0, 0.0, 0.0]
                for x in direct:
                    if x.endpoint.name.startswith(species + ", "):
                        assert x.principal_carbon_origin_fractions is not None
                        for i, fraction in enumerate(x.principal_carbon_origin_fractions):
                            attributed[i] += x.amount * 12 / molar * fraction
                assert attributed[0] / attributed[1] == pytest.approx(1.2 / 0.8)
                assert attributed[2] / attributed[1] == pytest.approx(0.5 / 0.8)
            phosphorus = (
                load.bottom_ash.waste_elements_kg["P"]
                + load.residual_ash.waste_elements_kg["P"]
                + emissions["Phosphorus", URBAN]
                + emissions["Phosphate", SURFACE] * 31 / 95
            )
            assert phosphorus == pytest.approx(load.material.elements_kg["P"])
            assert load.air_elements_kg["P"] == emissions["Phosphorus", URBAN]
            assert load.water_elements_kg["P"] == pytest.approx(
                emissions["Phosphate", SURFACE] * 31 / 95
            )
            assert load.bottom_ash.waste_elements_kg["P"] == pytest.approx(
                phosphorus * 0.8536803006664344
            )
            for fraction, ash, physical in (
                ("bottom", load.bottom_ash, result.bottom_ash),
                ("residual", load.residual_ash, result.residual_ash),
            ):
                received = [
                    x
                    for x in result.amounts
                    if x.function == f"pa_municipal_{fraction}_ash_treatment"
                ]
                p_release = fsum(
                    x.amount * 31 / 95 for x in received if x.endpoint.name == "Phosphate"
                )
                assert p_release + physical.retained_profile_elements_kg["P"] == pytest.approx(
                    ash.waste_elements_kg["P"]
                )
                market = [
                    x for x in result.amounts if x.function == f"pa_municipal_{fraction}_ash_market"
                ]
                assert len(market) == 3
                assert fsum(x.amount for x in market) == pytest.approx(
                    ash.treatment_mass_kg * 0.077
                )
                assert all(isinstance(x.endpoint, PaReceivingActivity) for x in market)
            # Existing hydrated cement is one complete negative waste demand.
            hydrated = [
                x
                for x in result.amounts
                if isinstance(x.endpoint, PaReceivingActivity)
                and x.endpoint.key[1] == "phb_hydrated"
            ]
            assert len(hydrated) == 1
            assert hydrated[0].amount == pytest.approx(-load.residual_ash.treatment_mass_kg)
