"""Follow finite PDO medium/cofactor elements through complete compiled CH receiving."""

import json
from collections import defaultdict
from dataclasses import replace
from math import fsum, log
from pathlib import Path
from typing import Any, cast

import pytest
from pa_water_receiving_fixtures import water_receiving_boundary

from pytoce.prepare_backgrounds.preparation.chemical_organic_wastewater import (
    chemical_organic_wastewater_datasets,
)
from pytoce.prepare_backgrounds.preparation.providers.ash_treatment import (
    _ash_market_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import provider_service_code
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _LeachateCarbon,
    _LeachateLoad,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_treatment import (
    _leachate_market_dataset,
    _sludge_market_dataset,
    _wastewater_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import (
    _scaled_native_exchange,
)
from pytoce.prepare_backgrounds.preparation.providers.swiss_sludge_incineration import (
    swiss_sludge_load,
)
from pytoce.prepare_backgrounds.preparation.providers.swiss_sludge_load import (
    ELEMENT_FUNCTIONS,
    FLUWA_SHARE,
)
from pytoce.prepare_backgrounds.preparation.providers.waste_gas import _wastewater_gas
from pytoce.prepare_backgrounds.preparation.providers.water_treatment import (
    _water_evaporation,
)


def _pdo_residual_inlet() -> _LeachateLoad:
    """Supply selected chemical identities at controlled test masses.

    Only the B12 dose is the adopted kg/kg-PDO prescription. Other masses
    deliberately exercise the selected medium's actual elements. Molecular
    H/O remain element masses, distinct from Doka's organic descriptors.
    """
    weights = {
        "C": 12.011,
        "H": 1.008,
        "O": 15.999,
        "N": 14.007,
        "P": 30.974,
        "Co": 58.933,
        "Ca": 40.078,
        "Cl": 35.45,
        "K": 39.0983,
        "Mg": 24.305,
        "S": 32.06,
        "Na": 22.98976928,
    }
    # Cyanocobalamin C63H88CoN14O14P; the two hydrated salts retain their
    # hydrate H/O. Invented non-cofactor quantities are test controls.
    materials = (
        (0.0000787697079936518, {"C": 63, "H": 88, "Co": 1, "N": 14, "O": 14, "P": 1}),
        (0.001, {"Ca": 1, "Cl": 2, "H": 4, "O": 2}),
        (0.002, {"K": 1, "H": 2, "P": 1, "O": 4}),
        (0.003, {"Mg": 1, "S": 1, "O": 11, "H": 14}),
        (0.0004, {"Na": 1, "O": 1, "H": 1}),
        (0.02, {"C": 6, "H": 12, "O": 6}),
    )
    elements: dict[str, float] = defaultdict(float)
    for mass, formula in materials:
        molar_mass = fsum(count * weights[element] for element, count in formula.items())
        for element, count in formula.items():
            elements[element] += mass * count * weights[element] / molar_mass
    assert fsum(elements.values()) == pytest.approx(fsum(mass for mass, _ in materials))
    # Only one cyanide carbon receives the qualified fossil attribution;
    # glucose and the 62 corrinoid carbons are nominal auxiliary biogenic C.
    fossil = materials[0][0] * weights["C"] / 1355.388
    carbon = elements["C"]
    return _LeachateLoad(
        dict(elements),
        (
            _LeachateCarbon("pdo_residual", False, "biogenic", carbon - fossil),
            _LeachateCarbon("pdo_b12_cyanide", False, "fossil", fossil),
        ),
        0.1225 * carbon,
        0.64 * carbon,
        carbon,
        0.68 * carbon,
        3.22 * carbon,
        1.7586552173908316 * carbon,
        chloride_kg=elements["Cl"],
    )


def _compiled_rows(
    datasets: list[tuple[dict[str, Any], float]], children: set[tuple[str, str]]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Expand each real negative-reference owner once, retaining external distributions."""
    rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for dataset, quantity in datasets:
        exchanges = dataset["exchanges"]
        production = next(row for row in exchanges if row["type"] == "production")
        multiplier = quantity / -production["amount"]
        for row in exchanges:
            if row["type"] != "production" and row["input"] not in children:
                rows[row["input"]].append(_scaled_native_exchange(row, multiplier))
    return dict(rows)


@pytest.mark.parametrize("future", (False, True))
def test_pdo_medium_metals_reach_water_soil_thermal_and_both_ashes(
    tmp_path: Path, future: bool
) -> None:
    """Compile positive PDO medium loads through their actual receiving owner.

    Original/future native portfolios have different, nonunit signed references
    and future properties are absent. Source calibration, complete supplies,
    source uncertainty and one collection fraction survive every receiving gate.
    This controlled composition tests the shared receiver; the actual assembled
    PDO factory residual is covered by its independent manufacturing test.
    """
    leachate = _pdo_residual_inlet()
    volume_m3, water_kg = 0.001, 0.95
    profile = _CH_WASTEWATER_PROFILE
    with water_receiving_boundary(tmp_path, future=future) as (bd, context):
        services = {
            str(service["provider_service_id"]): service
            for service in cast(list[dict[str, Any]], context.catalogue["services"])
        }
        ids = {
            role: f"pdo_fossil_{role}_ch"
            for role in (
                "water_collection",
                "water_treatment",
                "sludge_incineration_market",
                "sludge_incineration",
                "sludge_landfarming_market",
                "sludge_landfarming",
                "bottom_ash_market",
                "bottom_ash_treatment",
                "residual_ash_market",
                "residual_ash_treatment",
            )
        }
        originals = {
            role: context.source.activity(services[ids[role]]["construction"]["source"], year=2025)
            for role in (
                "water_treatment",
                "sludge_incineration",
                "sludge_landfarming",
                "bottom_ash_treatment",
                "residual_ash_treatment",
            )
        }
        original_rows = {
            node.key: json.dumps([dict(row) for row in node.exchanges()], sort_keys=True)
            for node in originals.values()
        }
        evaporation = _water_evaporation(
            str(originals["water_treatment"]["comment"]), profile.hydraulic
        )
        wastewater = _wastewater_load(
            leachate,
            volume_m3=volume_m3,
            incoming_water_kg=water_kg,
            evaporation_fraction=evaporation,
            profile=profile,
        )
        records = chemical_organic_wastewater_datasets(
            context,
            service_id=ids["water_collection"],
            load=leachate,
            solution_volume_m3=volume_m3,
            water_kg=water_kg,
            complete_carbon_oxidation=True,
            phosphorus_equilibrium=True,
        )
        target = context.provider_database
        collection = records[target, provider_service_code(ids["water_collection"])]
        production = next(row for row in collection["exchanges"] if row["type"] == "production")
        plant = next(
            row
            for row in collection["exchanges"]
            if row["input"] == (target, provider_service_code(ids["water_treatment"]))
        )
        treated = plant["amount"] / production["amount"]
        assert treated == pytest.approx(0.8)
        # Original avgWWT D/N rows36,57,59; biological/digestion gas are zero.
        # These independent oracles catch silent coordinate loss or an oxide
        # multiplier on kg element, before checking the compiled exchanges.
        for element, primary, total in (
            ("Co", 0.25770080368472997, 0.85900267894909987),
            ("Ca", 0.062112441519308538, 0.20704147173102846),
            ("K", 0.015895638998058619, 0.052985463326862062),
        ):
            mass = leachate.elements_kg[element]
            first = fsum(profile.hydraulic.stages) * primary * mass
            second = fsum(profile.hydraulic.stages[1:]) * (total - primary) * mass
            assert wastewater.primary_sludge.elements_kg[element] == pytest.approx(first)
            assert wastewater.secondary_sludge.elements_kg[element] == pytest.approx(second)
            assert wastewater.final_sludge.elements_kg[element] == pytest.approx(first + second)
            assert wastewater.biological_air_elements_kg[element] == 0
            assert wastewater.digester_gas_elements_kg[element] == 0
        for element in ("Co", "Ca", "K", "Mg", "Cl"):
            conditional = (
                wastewater.final_sludge.elements_kg[element]
                + wastewater.effluent.elements_kg[element]
            )
            assert conditional == pytest.approx(leachate.elements_kg[element])
            assert (1 - treated) * leachate.elements_kg[element] + treated * conditional == (
                pytest.approx(leachate.elements_kg[element])
            )
        assert wastewater.final_sludge.elements_kg["Cl"] == 0
        without_metals = _wastewater_load(
            replace(
                leachate,
                elements_kg={
                    element: mass
                    for element, mass in leachate.elements_kg.items()
                    if element not in {"Co", "Ca", "K"}
                },
            ),
            volume_m3=volume_m3,
            incoming_water_kg=water_kg,
            evaporation_fraction=evaporation,
            profile=profile,
        )
        # WWcalc FB36/57/59 applies to nominal primary dry-sludge oxygen.
        extra_oxygen = wastewater.primary_sludge.elements_kg["Co"] * (
            1.2715085694892245 - 1
        ) + wastewater.primary_sludge.elements_kg["Ca"] * (1.3992015968063873 - 1)
        assert wastewater.primary_sludge.allocated_oxygen_kg == pytest.approx(
            without_metals.primary_sludge.allocated_oxygen_kg + extra_oxygen
        )
        physical = swiss_sludge_load(
            context, services[ids["sludge_incineration"]]["construction"], wastewater
        )
        quantities = {
            "water_collection": volume_m3,
            "water_treatment": treated * volume_m3,
            "sludge_incineration_market": treated * physical.wet_mass_kg,
            "sludge_incineration": treated * physical.wet_mass_kg,
            "sludge_landfarming_market": treated * wastewater.landfarmed_sludge_wet_kg,
            "sludge_landfarming": treated * wastewater.landfarmed_sludge_wet_kg,
            "bottom_ash_market": treated * physical.bottom_ash.treatment_mass_kg,
            "bottom_ash_treatment": treated * physical.bottom_ash.treatment_mass_kg,
            "residual_ash_market": treated * physical.residual_ash.treatment_mass_kg,
            "residual_ash_treatment": treated * physical.residual_ash.treatment_mass_kg,
        }
        # Scale each negative functional unit once before checking elementary releases.
        compiled = {
            role: _compiled_rows(
                [(records[target, provider_service_code(ids[role])], quantity)], set(records)
            )
            for role, quantity in quantities.items()
        }
        flows = {
            key: (
                str(bd.get_node(database=key[0], code=key[1])["name"]),
                tuple(bd.get_node(database=key[0], code=key[1])["categories"]),
            )
            for rows in compiled.values()
            for key, exchanges in rows.items()
            if exchanges[0]["type"] == "biosphere"
        }
        for key, rows in original_rows.items():
            assert (
                json.dumps(
                    [dict(row) for row in bd.get_node(database=key[0], code=key[1]).exchanges()],
                    sort_keys=True,
                )
                == rows
            )

    # Original FLUWA Z/AB/AD rows36/57/59/60, on the kg-element extraction
    # inlet. Existing furnace transfers and technology share remain separate.
    for element, matrix, cake, liquid in (
        ("Co", 0.7587483233192104, 0.24118945653443066, 0.0000622201463589627),
        ("Ca", 0.8368600023988669, 0.08156999880056649, 0.08156999880056649),
        ("K", 0.557244444655638, 0.023608763992007138, 0.4191467913523548),
        ("Mg", 0.9145217448313052, 0.027418618269793263, 0.05805963689890142),
    ):
        mass = physical.elements_kg[element]
        combustion = ELEMENT_FUNCTIONS[element]
        extraction = FLUWA_SHARE * mass * fsum(combustion[1:5])
        assert physical.phases["bottom"][element] == pytest.approx(
            mass * combustion[0] + extraction * matrix
        )
        assert physical.phases["hydroxide"][element] == pytest.approx(extraction * cake)
        assert physical.phases["water"][element] == pytest.approx(
            (1 - FLUWA_SHARE) * mass * combustion[4] + extraction * liquid
        )
    for element, name in (
        ("Co", "Cobalt II"),
        ("Ca", "Calcium II"),
        ("K", "Potassium I"),
        ("Mg", "Magnesium"),
    ):
        raw = [
            row
            for key, rows in compiled["water_collection"].items()
            if flows.get(key) == (name, ("water",))
            for row in rows
        ]
        assert fsum(row["amount"] for row in raw) == pytest.approx(
            (1 - treated) * leachate.elements_kg[element]
        )
        effluent = [
            row
            for key, rows in compiled["water_treatment"].items()
            if flows.get(key) == (name, ("water", "surface water"))
            for row in rows
        ]
        assert fsum(row["amount"] for row in effluent) == pytest.approx(
            treated * wastewater.effluent.elements_kg[element]
        )
        assert physical.elements_kg[element] == pytest.approx(
            wastewater.incineration_dry_share * wastewater.final_sludge.elements_kg[element]
        )
        assert fsum(phase[element] for phase in physical.phases.values()) == pytest.approx(
            physical.elements_kg[element]
        )
        for categories, phase in (
            (("air", "urban air close to ground"), "air"),
            (("water", "surface water"), "water"),
        ):
            assert physical.phases[phase][element] > 0
            emitted = [
                row
                for key, rows in compiled["sludge_incineration"].items()
                if flows.get(key) == (name, categories)
                for row in rows
            ]
            assert emitted
            assert fsum(row["amount"] for row in emitted) == pytest.approx(
                treated * physical.emissions[name, "kilogram", categories]
            )
        for ash in (physical.bottom_ash, physical.residual_ash):
            assert ash.waste_elements_kg[element] > 0
        for role in (
            "sludge_landfarming",
            "sludge_incineration",
            "bottom_ash_treatment",
            "residual_ash_treatment",
        ):
            releases = [
                row
                for key, rows in compiled[role].items()
                if key in flows and flows[key][0] == name
                for row in rows
            ]
            assert releases
            for row in releases:
                # The invented native source assigns the same lognormal
                # offset to these element rows; scaling preserves it.
                assert row["amount"] > 0
                assert row["uncertainty type"] == 2
                assert row["scale"] == 0.21
                assert row["loc"] == pytest.approx(log(row["amount"]) + 0.03)
    assert not any(
        flows.get(key, (None, ()))[0] == "Chlorine" for key in compiled["water_collection"]
    )
    raw_chloride = [
        row
        for key, rows in compiled["water_collection"].items()
        if flows.get(key) == ("Chloride", ("water", "surface water"))
        for row in rows
    ]
    assert fsum(row["amount"] for row in raw_chloride) == pytest.approx(
        (1 - treated) * leachate.chloride_kg
    )


@pytest.mark.parametrize("original_physical_reference", (False, True))
def test_ch_receiving_selects_fixed_physical_coefficients_and_dated_supplies(
    tmp_path: Path, original_physical_reference: bool
) -> None:
    """Separate CH references from changed dated ratios and gas factors.

    Only invented external future rows are varied. The default remains dated;
    CH manufacture instead retains original mass handoffs, embodied electricity
    and matched gas/fuel factors, with complete dated sewer and haul suppliers.
    """
    leachate = _pdo_residual_inlet()
    profile = _CH_WASTEWATER_PROFILE
    volume_m3, water_kg = 0.001, 0.95
    market_roles = (
        "water_collection",
        "sludge_incineration_market",
        "sludge_landfarming_market",
        "bottom_ash_market",
        "residual_ash_market",
    )
    with water_receiving_boundary(tmp_path, future=True) as (bd, context):
        services = {row["provider_service_id"]: row for row in context.catalogue["services"]}
        selected = {
            role: services[f"pdo_fossil_{role}_ch"]
            for role in (*market_roles, "water_treatment", "sludge_incineration")
        }
        originals = {
            role: context.source.activity(service["construction"]["source"], year=2025)
            for role, service in selected.items()
        }
        dated = {
            role: context.ordinary.activity(service["construction"]["source"], year=2050)
            for role, service in selected.items()
        }
        modified = []
        for role in market_roles:
            product = selected[role]["construction"]["source"]["reference_product"]
            modified.extend(
                (row, 0.5)
                for row in dated[role].technosphere()
                if row.input["reference product"] == product
            )
        modified.extend(
            (row, 3.0)
            for row in dated["water_treatment"].technosphere()
            if row.input["reference product"] == "limestone, crushed, washed"
        )
        gas_sources = selected["water_treatment"]["construction"]["gas_sources"]
        original_gas = {
            role: context.source.activity(selector, year=2025)
            for role, selector in gas_sources.items()
        }
        for role, product in (("chp", "biogas"), ("furnace", "natural gas, high pressure")):
            activity = context.ordinary.activity(gas_sources[role], year=2050)
            modified.extend(
                (row, 2.0)
                for row in activity.technosphere()
                if row.input["reference product"] == product
            )
            modified.extend((row, 3.0) for row in activity.biosphere())
        # These independent fixture changes distinguish a dated production
        # basis from coefficients; uncertainty still describes each row.
        for row, factor in modified:
            row["amount"] *= factor
            if "loc" in row:
                row["loc"] = (
                    row["loc"] + log(factor)
                    if row.get("uncertainty type") == 2
                    else row["loc"] * factor
                )
            for field in ("minimum", "maximum"):
                if field in row:
                    row[field] *= factor
            row.save()
        evaporation = _water_evaporation(originals["water_treatment"]["comment"], profile.hydraulic)
        wastewater = _wastewater_load(
            leachate,
            volume_m3=volume_m3,
            incoming_water_kg=water_kg,
            evaporation_fraction=evaporation,
            profile=profile,
        )
        physical = swiss_sludge_load(
            context, selected["sludge_incineration"]["construction"], wastewater
        )
        gas = _wastewater_gas(
            context,
            selected["water_treatment"]["construction"],
            wastewater,
            selected["water_treatment"]["provider_service_id"],
            profile,
            original_physical_reference=original_physical_reference,
        )
        target = context.provider_database
        records = {
            "water_collection": _leachate_market_dataset(
                context,
                selected["water_collection"],
                leachate,
                wastewater,
                target,
                original_physical_reference=original_physical_reference,
            ),
            "water_treatment": _wastewater_dataset(
                context,
                selected["water_treatment"],
                leachate,
                wastewater,
                gas,
                target,
                profile,
                original_physical_reference=original_physical_reference,
            ),
        }
        for role in ("sludge_incineration_market", "sludge_landfarming_market"):
            records[role] = _sludge_market_dataset(
                context,
                selected[role],
                wastewater,
                target,
                original_physical_reference=original_physical_reference,
            )
        for role, load in (
            ("bottom_ash_market", physical.bottom_ash),
            ("residual_ash_market", physical.residual_ash),
        ):
            records[role] = _ash_market_dataset(
                context,
                selected[role],
                load,
                target,
                original_physical_reference=original_physical_reference,
            )
        for role in market_roles:
            output = records[role]["exchanges"]
            production = next(row["amount"] for row in output if row["type"] == "production")
            dated_production = next(row["amount"] for row in dated[role].production())
            assert production == dated_production < 0
            product = selected[role]["construction"]["source"]["reference_product"]
            basis = originals[role] if original_physical_reference else dated[role]
            source_production = next(row["amount"] for row in basis.production())
            terminal = [
                row for row in basis.technosphere() if row.input["reference product"] == product
            ]
            child = [
                row for row in output if row["type"] == "technosphere" and row["input"][0] == target
            ]
            assert len(child) == len(terminal)
            assert fsum(row["amount"] for row in child) / production == pytest.approx(
                fsum(row["amount"] for row in terminal) / source_production
            )
            for bound, source in zip(child, terminal, strict=True):
                factor = production / source_production
                assert bound["amount"] == pytest.approx(source["amount"] * factor)
                assert bound["uncertainty type"] == source["uncertainty type"]
                expected_loc = (
                    source["loc"] + log(factor)
                    if source["uncertainty type"] == 2
                    else source["loc"] * factor
                )
                assert bound["loc"] == pytest.approx(expected_loc)
            # Complete sewer/haul rows retain the future source and its native
            # amounts/distributions. No original supplier is purchased instead.
            retained = {
                row["input"]: row
                for row in output
                if row["type"] == "technosphere" and row["input"][0] != target
            }
            expected = [
                row
                for row in dated[role].technosphere()
                if row.input["reference product"] != product
            ]
            assert retained.keys() == {row["input"] for row in expected}
            for row in expected:
                actual = retained[row["input"]]
                for field in ("amount", "uncertainty type", "loc", "scale", "minimum", "maximum"):
                    if field in row:
                        assert actual[field] == row[field]
        collection_production = next(
            row["amount"]
            for row in records["water_collection"]["exchanges"]
            if row["type"] == "production"
        )
        treated = 0.8 if original_physical_reference else 0.4
        bypass = fsum(
            row["amount"]
            for row in records["water_collection"]["exchanges"]
            if row["type"] == "biosphere" and bd.get_node(key=row["input"])["name"] == "Chloride"
        )
        assert bypass == pytest.approx(
            -collection_production * (1 - treated) / volume_m3 * leachate.chloride_kg
        )
        basis = (
            originals["water_treatment"]
            if original_physical_reference
            else dated["water_treatment"]
        )
        p0 = next(row["amount"] for row in basis.production())
        p = next(row["amount"] for row in dated["water_treatment"].production())
        limestone = fsum(
            row["amount"]
            for row in basis.technosphere()
            if row.input["reference product"] == "limestone, crushed, washed"
        )
        scale = -p / volume_m3
        gross = scale * wastewater.duties.electricity_kwh + p / p0 * limestone * 23375 / 13250
        electricity = fsum(
            row["amount"]
            for row in records["water_treatment"]["exchanges"]
            if row["type"] == "technosphere"
            and row["input"][0] != target
            and bd.get_node(key=row["input"])["reference product"] == "electricity, medium voltage"
        )
        assert electricity == pytest.approx(gross - scale * gas.internal_electricity_kwh)
        heat = fsum(
            row["amount"]
            for row in records["water_treatment"]["exchanges"]
            if row["type"] == "biosphere"
            and bd.get_node(key=row["input"])["name"] == "Heat, waste"
            and bd.get_node(key=row["input"])["categories"][0] == "air"
        )
        assert heat == pytest.approx(
            gross * 3.6
            + scale * (gas.heat_air_mj + wastewater.duties.heat_mj - gas.internal_heat_mj)
        )
        fuels, factors, lhv = {}, {}, {}
        for role, product in (("chp", "biogas"), ("furnace", "natural gas, high pressure")):
            fuel = next(
                row
                for row in original_gas[role].technosphere()
                if row.input["reference product"] == product
            )
            fuels[role] = fuel["amount"]
            lhv[role] = fuel["properties"]["heating value, net"]["amount"]
            factors[role] = fsum(
                row["amount"]
                for row in original_gas[role].biosphere()
                if row.input["name"] == "Dinitrogen monoxide"
            )
        # Future synthetic fuel doubles and emissions triple; original mode retains original-source
        # values for both.
        factor_change = 1.0 if original_physical_reference else 1.5
        gas_n2o = fsum(
            row["amount"]
            for row in gas.biosphere_exchanges
            if bd.get_node(key=row["input"])["name"] == "Dinitrogen monoxide"
        )
        assert gas_n2o == pytest.approx(
            gas.trace["raw_dry_normal_volume_m3"]
            * (1 - 0.0075)
            * factor_change
            * (
                profile.chp_share * factors["chp"] / fuels["chp"]
                + (1 - profile.chp_share)
                * lhv["chp"]
                / lhv["furnace"]
                * factors["furnace"]
                / fuels["furnace"]
            )
        )
        terms = [
            json.loads(row["project_terms_json"])
            for row in context.technology_rows
            if row.get("project_terms_json")
        ]
        references = [
            row["physical_reference_source"] for row in terms if "physical_reference_source" in row
        ]
        assert len(references) == (8 if original_physical_reference else 0)
        assert all(key[0] == "original" for key in references)
