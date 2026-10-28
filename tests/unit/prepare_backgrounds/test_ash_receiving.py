"""Test actual ash releases and PA phosphorus through complete native receivers.

Inputs are small invented ash loads and the existing unlicensed source graph.
The source short/cumulative P constants below are independent test authorities;
their kg-P fractions must not be treated as already speciated phosphate kg.
"""

import json
from math import fsum, log
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from bw2calc import LCA
from numpy.typing import NDArray
from pa_landfill_receiving_fixtures import landfill_receiving_boundary

from pytoce.prepare_backgrounds.preparation.providers.ash_treatment import (
    _PA_ASH_RELEASES,
    AshSource,
    _ash_dataset,
    _ash_market_dataset,
    _AshLoad,
    _TreatmentCarbon,
    calculate_ash_receiving,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    provider_service_code,
)

SURFACE = ("water", "surface water")
LATER = ("water", "ground-, long-term")
P_FRACTIONS = {
    "bottom": (0.000022037531153838305, 0.013222518692302985),
    "residual": (0.00037317372036873187, 0.22390423222123912),
}


def _load(phosphorus: float) -> _AshLoad:
    """Supply a small internally balanced wet ash with separate waste/APC element masses."""
    waste = {"C": 0.1, "N": 0.02, "S": 0.01, "Zn": 0.005, "P": phosphorus, "Na": 0.0}
    auxiliary = {"C": 0.002, "Ca": 0.003, "Ti": 0.001, "V": 0.0001}
    water, hydrogen, oxygen = 0.1, 0.02, 0.3
    return _AshLoad(
        treatment_mass_kg=fsum((*waste.values(), *auxiliary.values(), water, hydrogen, oxygen)),
        water_kg=water,
        waste_elements_kg=waste,
        allocated_hydrogen_kg=hydrogen,
        allocated_oxygen_kg=oxygen,
        auxiliary_elements_kg=auxiliary,
        carbon=(
            _TreatmentCarbon("principal", True, "atmospheric_dac", 0.1),
            _TreatmentCarbon("PAM", False, "fossil", 0.002),
        ),
    )


def _source() -> AshSource:
    """Use invented original coefficients with absent P and separate native rows."""
    return AshSource(
        dry_fraction=0.8,
        element_contents={"C": 0.4, "N": 0.04, "S": 0.02, "Zn": 0.0088},
        emissions=(
            ("DOC, Dissolved Organic Carbon", SURFACE, 1e-5),
            ("DOC, Dissolved Organic Carbon", SURFACE, 2e-5),
            ("DOC, Dissolved Organic Carbon", LATER, 0.0),
            ("Nitrate", SURFACE, 1e-7),
            ("Nitrate", LATER, 2e-7),
            ("Sulfate", SURFACE, 3e-7),
            ("Sulfate", LATER, 4e-7),
            ("Zinc II", SURFACE, 5e-7),
            ("Zinc II", LATER, 6e-7),
            ("Non-hazardous waste disposed", SURFACE, 1.0),
            ("Sodium I", SURFACE, 1e-9),
            ("Cadmium II", SURFACE, 1e-10),
            # The synthetic donor trace-metal flows remain separate from actual PA ash composition.
            ("Scandium", SURFACE, 1e-10),
            ("Silver I", SURFACE, 1e-10),
            ("Tungsten", SURFACE, 1e-10),
            ("Bromine", LATER, 1e-10),
            ("Chloride", LATER, 1e-10),
            ("Fluoride", LATER, 1e-10),
        ),
    )


@pytest.mark.parametrize("fraction", ("bottom", "residual"))
@pytest.mark.parametrize("phosphorus", (0.0, 0.0007), ids=("no-P", "actual-P"))
def test_actual_phosphorus_and_existing_source_functions_close(fraction, phosphorus):
    """Close added elemental P and preserve every supplied native release factor."""
    load = _load(phosphorus)
    source = _source()
    result = calculate_ash_receiving(load, source, release_profile=_PA_ASH_RELEASES[fraction])
    short, cumulative = P_FRACTIONS[fraction]
    added = {(name, compartment): amount for name, compartment, amount in result.additions}
    assert added["Phosphate", SURFACE] == pytest.approx(phosphorus * short * 95 / 31)
    assert added["Phosphate", LATER] == pytest.approx(phosphorus * (cumulative - short) * 95 / 31)
    assert result.retained_profile_elements_kg["P"] == pytest.approx(phosphorus * (1 - cumulative))
    assert (
        (added["Phosphate", SURFACE] + added["Phosphate", LATER]) * 31 / 95
        + result.retained_profile_elements_kg["P"]
    ) == pytest.approx(phosphorus)
    assert len([name for name, _, _ in result.additions if name == "Phosphate"]) == 2

    native = {row.source_index: row for row in result.native}
    # Repeated rows retain individual identities. Valid native zero amounts retain their scale.
    carbon_scale = 0.102 / (0.8 * 0.4)
    for index, amount in ((0, 1e-5), (1, 2e-5), (2, 0.0)):
        assert native[index].amount == pytest.approx(amount * carbon_scale, rel=2e-12, abs=0)
        assert native[index].multiplier == pytest.approx(carbon_scale)
    assert native[9].amount == load.treatment_mass_kg
    # Zero sodium and elements absent from the actual load receive no donor release.
    assert set(native).isdisjoint(range(10, len(source.emissions)))
    # Each row retains its source coefficient and time window. Nitrate/sulfate are kg compound.
    for element, indices, coefficients, content in (
        ("N", (3, 4), (1e-7, 2e-7), 0.04),
        ("S", (5, 6), (3e-7, 4e-7), 0.02),
        ("Zn", (7, 8), (5e-7, 6e-7), 0.0088),
    ):
        incoming = load.waste_elements_kg[element]
        for index, coefficient in zip(indices, coefficients, strict=True):
            assert native[index].amount == pytest.approx(coefficient * incoming / (0.8 * content))
    if fraction == "bottom":
        assert set(result.retained_profile_elements_kg) == {"P"}
    else:
        assert added["Calcium II", SURFACE] == pytest.approx(0.003 * 0.0001508)
        assert added["Calcium II", LATER] == pytest.approx(0.003 * (0.09051 - 0.0001508))
        assert result.retained_profile_elements_kg["Ca"] == pytest.approx(0.003 * (1 - 0.09051))


@pytest.mark.parametrize("future", (False, True), ids=("original", "future"))
def test_actual_pa_ash_uses_complete_signed_native_receiver(tmp_path: Path, future: bool):
    """Bind actual P once while preserving original chemistry and all dated support."""
    with landfill_receiving_boundary(tmp_path, future=future) as (bd, context):
        services = {
            str(row["provider_service_id"]): row
            for row in cast(list[dict[str, object]], context.catalogue["services"])
        }
        provider = "actual_ash_receiving"
        database = "future" if future else "original"
        records = {}
        payloads = {"bottom": _load(0.0007), "residual": _load(0.0013)}
        for fraction, load in payloads.items():
            for role, compiler in (("market", _ash_market_dataset), ("treatment", _ash_dataset)):
                sid = f"sbr_filled_fossil_{fraction}_ash_{role}"
                kwargs = (
                    {"release_profile": _PA_ASH_RELEASES[fraction]} if role == "treatment" else {}
                )
                records[provider, provider_service_code(sid)] = compiler(
                    context, services[sid], load, provider, **kwargs
                )
        bd.Database(provider).write(records, searchable=False)
        demand = {
            bd.get_node(
                database=provider,
                code=provider_service_code(f"sbr_filled_fossil_{fraction}_ash_market"),
            ): -load.treatment_mass_kg
            for fraction, load in payloads.items()
        }
        calculation = LCA(demand)
        calculation.lci()
        supply_array = cast(NDArray[np.float64], calculation.supply_array)
        for fraction, load in payloads.items():
            sid = f"sbr_filled_fossil_{fraction}_ash_treatment"
            terminal = bd.get_node(database=provider, code=provider_service_code(sid))
            original = bd.get_node(database="original", code=f"thermal_{fraction}")
            dated = bd.get_node(database=database, code=f"thermal_{fraction}")
            original_properties = original.rp_exchange()["properties"]
            assert "phosphorus content" not in original_properties
            assert not any(ex.input["name"] == "Phosphate" for ex in original.biosphere())
            if future:
                assert "properties" not in dated.rp_exchange()
            production = terminal.rp_exchange()["amount"]
            assert production == (-7 if future else -3)
            scale = -production / load.treatment_mass_kg
            phosphate = {
                tuple(ex.input["categories"]): ex["amount"] / scale
                for ex in terminal.biosphere()
                if ex.input["name"] == "Phosphate"
            }
            short, cumulative = P_FRACTIONS[fraction]
            incoming = load.waste_elements_kg["P"]
            assert phosphate == pytest.approx(
                {
                    SURFACE: incoming * short * 95 / 31,
                    LATER: incoming * (cumulative - short) * 95 / 31,
                }
            )
            assert len([ex for ex in terminal.biosphere() if ex.input["name"] == "Phosphate"]) == 2
            terms = json.loads(
                cast(
                    str,
                    next(
                        row["project_terms_json"]
                        for row in context.technology_rows
                        if row.get("provider_service_id") == sid
                        and row.get("component_id") == "ash_terminal"
                    ),
                )
            )
            assert terms["explicit_profile_retained_elements_kg"]["P"] == pytest.approx(
                incoming * (1 - cumulative)
            )
            # Every original Ti/V row keeps its own intensive calibration and
            # lognormal metadata. The CH/TPS profiles have separate regressions.
            for name, element, property_name in (
                ("Titanium ion", "Ti", "titanium content"),
                ("Vanadium V", "V", "vanadium content"),
            ):
                for ex in terminal.biosphere():
                    if ex.input["name"] != name:
                        continue
                    source = next(
                        row for row in original.biosphere() if row.input.key == ex.input.key
                    )
                    coefficient = source["amount"] / 3
                    expected = (
                        coefficient
                        * load.auxiliary_elements_kg[element]
                        / (0.8 * original_properties[property_name]["amount"])
                    )
                    assert ex["amount"] / scale == pytest.approx(expected, rel=2e-12)
                    assert ex["scale"] == source["scale"]
                    assert ex["loc"] == pytest.approx(log(ex["amount"]) + 0.03)
                    assert ex["allocation"] == source["allocation"]
            assert len(list(terminal.technosphere())) == len(list(dated.technosphere()))
            for source in dated.technosphere():
                actual = next(
                    ex for ex in terminal.technosphere() if ex.input.key == source.input.key
                )
                for field in ("amount", "loc", "uncertainty type", "allocation"):
                    assert actual[field] == source[field]
            # The matrix solve checks signed market/terminal production and freight normalization
            # applied once.
            assert supply_array[calculation.dicts.activity[terminal.id]] == pytest.approx(
                load.treatment_mass_kg / -production, rel=2e-7
            )
        residual = bd.get_node(
            database=provider,
            code=provider_service_code("sbr_filled_fossil_residual_ash_treatment"),
        )
        assert {ex.input["code"] for ex in residual.technosphere()} == {
            "thermal_support",
            "thermal_cement",
            "thermal_hydration",
            "thermal_hydrated",
        }
