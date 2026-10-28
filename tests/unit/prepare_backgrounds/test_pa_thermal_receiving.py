"""Keep complete native PA maintenance duties after portable source admission."""

import json
from dataclasses import dataclass, replace
from math import fsum, log
from typing import cast

import pytest
from chemical_residue_fixtures import pa_maintenance_context

from pytoce.prepare_backgrounds.preparation.chemical_residue_incineration import (
    CHEMICAL_RESIDUE_ASH_SOURCE,
    CHEMICAL_RESIDUE_FIXED_SOURCE,
    CHEMICAL_RESIDUE_PAM_SOURCE,
    CHEMICAL_RESIDUE_SOURCE,
    _chemical_residue_ash_source,
    _chemical_residue_source,
    _native_treatment,
)
from pytoce.prepare_backgrounds.preparation.chemical_residue_load import _residue_reference
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_residues import (
    _pa_maintenance_residue_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
    _activity,
    _flow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
    _operation,
    reference_from_payload,
    reference_to_payload,
    sources_from_payload,
    sources_to_payload,
)
from pytoce.run_lca.calculation.pa_receiving import PaSpinningEvent
from pytoce.run_lca.calculation.pa_thermal_receiving import receive_pa_maintenance
from tests.pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources

_CASES = (
    ("pa6", (0.4, 0.35, 0.25), 0.0000075),
    ("pa66", (0.4, 0.35, 0.25), 0.0000075),
    ("pa6", (1.0, 0.0, 0.0), 0.0),
    ("pa6", (0.0, 1.0, 0.0), 0.0),
    ("pa6", (0.0, 0.0, 1.0), 0.0),
)


@dataclass(frozen=True)
class _ThermalBoundary:
    """Store real thermal admission after leaving its original/future project."""

    reference: PaReceivingReference
    sources: PaReceivingSources
    events: tuple[PaSpinningEvent, ...]


@pytest.fixture(scope="module", params=[True, False], ids=["original", "future"])
def thermal_boundary(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> _ThermalBoundary:
    """Admit complete thermal rows, then discard all live project handles.

    Serialized records supply the gas, Swiss wastewater, and landfill sections.
    This test evaluates the thermal section. Separate receiving tests cover source
    selection. The shared finish calculation supplies the retained seven-ingredient mixture.
    """
    original = bool(request.param)
    root = tmp_path_factory.mktemp("detached_pa_thermal")
    events = []
    with pa_maintenance_context(root, original=original) as (_, context, database):
        hwi = _native_treatment(context, CHEMICAL_RESIDUE_SOURCE, "actual_pa_thermal_test")
        fixed = _native_treatment(context, CHEMICAL_RESIDUE_FIXED_SOURCE, "actual_pa_thermal_test")
        ash = _native_treatment(context, CHEMICAL_RESIDUE_ASH_SOURCE, "actual_pa_thermal_test")
        reheat = (
            "heat, district or industrial, natural gas"
            if original
            else "heat, district or industrial"
        )
        source = _chemical_residue_source(
            context, hwi, fixed, CHEMICAL_RESIDUE_SOURCE, reheat, integrated_pa_site=True
        )
        ash_source = _chemical_residue_ash_source(context, ash, CHEMICAL_RESIDUE_ASH_SOURCE)
        reference = replace(
            pa_receiving_reference(),
            hwi=_activity(context.source.activity(CHEMICAL_RESIDUE_SOURCE, year=2025)),
            donor_elements_kg=source.donor_elements_kg,
            ash=_activity(context.source.activity(CHEMICAL_RESIDUE_ASH_SOURCE, year=2025)),
            ash_dry_fraction=ash_source.dry_fraction,
            ash_element_contents=ash_source.element_contents,
        )
        detached = pa_receiving_sources()
        pam, _ = context.activity(
            CHEMICAL_RESIDUE_PAM_SOURCE,
            source_role="native_ecoinvent_provider",
            service_id="actual_pa_thermal_test",
            component_id="complete_PAM_supply",
            semantic_class="service_normalization",
        )
        sources = replace(
            detached,
            background_scenario=context.background_scenario,
            provider_anchor_year=context.year,
            general_background_database=database,
            general_background_anchor_year=context.year,
            hwi=_operation(context, hwi, "hwi", reheat),
            fixed=_operation(context, fixed, "fixed", reheat),
            ash=_operation(context, ash, "ash", reheat),
            reheat_product=reheat,
            pam=_activity(pam),
            # Spinning's organic-air release has separate source-admission
            # coverage; this thermal fixture binds only maintenance outputs.
            added_biosphere=tuple(
                _flow(
                    context.biosphere.flow(
                        {"name": flow.name, "unit": flow.unit, "categories": list(flow.categories)}
                    )
                )
                for flow in detached.added_biosphere
                if flow.name != "NMVOC, non-methane volatile organic compounds"
            ),
        )
        for grade, shares, embedded in _CASES:
            # The fixture supplies destroyed principal and finish. A non-unit finished output checks
            # that receiving uses those quantities directly.
            principal = cast(tuple[float, float, float], tuple(0.00002 * x for x in shares))
            finished = 2.5
            event = PaSpinningEvent(
                year=context.year,
                region="NEU",
                constituent={"pa6": "polyamide_6_resin", "pa66": "polyamide_66_resin"}[grade],
                finished_principal_kg=finished,
                gross_principal_kg=finished + sum(principal),
                recovered_principal_kg=0.5,
                embedded_finish_dose_kg=0.001,
                maintenance_principal_by_origin_kg=principal,
                maintenance_finish_dose_kg=embedded,
                finished_principal_carbon_by_origin_kg=cast(
                    tuple[float, float, float], tuple(finished * x * 72 / 113 for x in shares)
                ),
                caprolactam_air_by_origin_kg=(0.0, 0.0, 0.0),
                caprolactam_water_by_origin_kg=(0.0, 0.0, 0.0),
            )
            events.append(event)
        # The receiving calculation consumes prepared records transported through strict JSON.
        reference = reference_from_payload(
            json.loads(json.dumps(reference_to_payload(reference), allow_nan=False))
        )
        sources = sources_from_payload(
            json.loads(json.dumps(sources_to_payload(sources), allow_nan=False))
        )
    return _ThermalBoundary(reference, sources, tuple(events))


@pytest.mark.parametrize(
    "case",
    range(len(_CASES)),
    ids=["pa6_mixed", "pa66_mixed", "fossil", "biogenic", "dac"],
)
def test_detached_maintenance_preserves_actual_complete_native_receiving(
    thermal_boundary: _ThermalBoundary, case: int
) -> None:
    """Bind native rows, original/future support, and origins from stored records.

    Expectations use the invented native source coefficients and actual input pools. The expected
    inventory follows the fixture source equations.
    """
    boundary = thermal_boundary
    event = boundary.events[case]
    grade, _, embedded = _CASES[case]
    load = _pa_maintenance_residue_load(
        _pa_finish_reference(grade),
        principal_by_origin_kg=event.maintenance_principal_by_origin_kg,
        embedded_dose_kg=embedded,
    )
    result = receive_pa_maintenance(event, boundary.reference, boundary.sources)
    mass, elements = load.wet_mass_kg, load.active_elements_kg
    assert result.site_water_kg == pytest.approx(5 * mass)
    database = boundary.sources.general_background_database
    supplies: dict[tuple[str, str], float] = {}
    emitted: dict[tuple[str, tuple[str, ...]], float] = {}
    terminal: dict[tuple[str, tuple[str, ...]], float] = {}
    ash_supplies: dict[tuple[str, str], float] = {}
    for row in result.amounts:
        if isinstance(row.endpoint, PaReceivingActivity):
            target = ash_supplies if row.function == "pa_maintenance_ash" else supplies
            target[row.endpoint.key] = target.get(row.endpoint.key, 0.0) + row.amount
        else:
            target = terminal if row.function == "pa_maintenance_ash" else emitted
            key = row.endpoint.name, row.endpoint.categories
            target[key] = target.get(key, 0.0) + row.amount
    assert not {
        (database, code)
        for code in (
            "mc_ng",
            "mc_oil",
            "mc_hwi_facility",
            "mc_fixed",
            "cr_road",
            "cr_rail",
            "mc_inorganic",
        )
    }.intersection(supplies)
    # The pure owner's independently tested chemistry supplies duties;
    # this boundary must bind each one to its dated native supplier exactly.
    physical = _residue_reference(
        load, boundary.sources.thermal_source(boundary.reference), integrated_pa_site=True
    )
    for code, duty in (
        ("mc_hcl", "HCl"),
        ("mc_naoh", "NaOH"),
        ("cr_peroxide", "H2O2"),
        ("cr_calcium_chloride", "CaCl2"),
        ("mc_cao", "CaO"),
        ("mc_pam", "PAM"),
    ):
        assert supplies.get((database, code), 0.0) == pytest.approx(
            physical["supplies"][duty], rel=2e-12, abs=1e-30
        )
    # Fixed catalyst follows wet mass; waste-dependent supply follows actual N.
    nr = elements["N"] / 0.01
    assert supplies.get((database, "mc_ammonia"), 0.0) == pytest.approx(0.002 * mass + 0.0001 * nr)
    assert supplies.get((database, "mc_tio2"), 0.0) == pytest.approx(0.0002 * mass + 0.0001 * nr)
    assert supplies.get((database, "mc_v_proxy"), 0.0) == pytest.approx(
        0.00005 * mass + 0.00002 * nr
    )
    urban, surface = ("air", "urban air close to ground"), ("water", "surface water")
    assert ("Water, well, in ground", ("natural resource", "in water")) not in emitted
    formed = load.waste_formed_water_kg + 27 / 17 * (
        supplies[database, "mc_ammonia"] - emitted["Ammonia", urban]
    )
    assert emitted["Water", ("air",)] * 1000 == pytest.approx(0.5 * mass + formed)
    assert emitted["Water", ("water",)] * 1000 == pytest.approx(4.5 * mass)
    assert emitted["Heat, waste", urban] == pytest.approx(0.83 * load.net_energy_mj)
    assert emitted["Heat, waste", surface] == pytest.approx(0.17 * load.net_energy_mj)
    assert emitted["Hazardous waste disposed", ("inventory indicator", "waste")] == pytest.approx(
        mass
    )
    # Every fixed source row survives except the site's reheat/water allocation.
    fixed_rows = [row for row in result.amounts if row.source == boundary.sources.fixed.activity]
    for native in boundary.sources.fixed.rows:
        endpoint = native.endpoint
        if isinstance(endpoint, PaReceivingActivity):
            if endpoint.reference_product == boundary.sources.reheat_product:
                continue
        elif endpoint.name in {"Water", "Water, well, in ground"}:
            continue
        matches = [row for row in fixed_rows if row.endpoint.key == endpoint.key]
        assert len(matches) == 1
        assert matches[0].amount == pytest.approx(native.amount * mass)
    ash_mass = terminal["Waste mass, total, placed in landfill", ("inventory indicator", "waste")]
    assert ash_mass > 0
    assert ash_supplies[database, "cr_road"] == pytest.approx(0.025 * ash_mass)
    assert ash_supplies[database, "mc_ash_support_a"] == pytest.approx(
        (0.25 if database == "original" else 0.8) * ash_mass
    )
    assert ash_supplies[database, "mc_ash_support_b"] == pytest.approx(
        (0.75 if database == "original" else 0.2) * ash_mass
    )
    assert ash_supplies[database, "mc_cement_a"] + ash_supplies[
        database, "mc_cement_b"
    ] == pytest.approx(0.4 * ash_mass)
    assert ash_supplies[database, "mc_hydrated_cement"] == pytest.approx(-ash_mass)
    assert ash_supplies[database, "mc_tap_a"] == pytest.approx(0.6 * ash_mass)
    for native in boundary.sources.ash.rows:
        if isinstance(native.endpoint, PaReceivingActivity):
            matches = [
                row
                for row in result.amounts
                if row.function == "pa_maintenance_ash" and row.endpoint.key == native.endpoint.key
            ]
            assert len(matches) == 1
            assert matches[0].amount == pytest.approx(native.amount * ash_mass)
    air_carbon = fsum(
        emitted.get((name, urban), 0.0) * fraction
        for name, fraction in (
            ("Carbon dioxide, fossil", 12 / 44),
            ("Carbon dioxide, non-fossil", 12 / 44),
            ("Carbon monoxide, fossil", 12 / 28),
            ("Carbon monoxide, non-fossil", 12 / 28),
            ("Cyanide", 12 / 26),
        )
    )
    assert air_carbon == pytest.approx(0.9889898 * elements["C"])
    fresh_carbon = supplies[database, "mc_pam"] * 36 / 71
    assert air_carbon + emitted["DOC, Dissolved Organic Carbon", surface] + terminal[
        "Organic carbon, placed in landfill", ("inventory indicator", "waste")
    ] == pytest.approx(elements["C"] + fresh_carbon)
    for gas in ("Carbon dioxide", "Carbon monoxide"):
        total = emitted.get((gas + ", fossil", urban), 0.0) + emitted.get(
            (gas + ", non-fossil", urban), 0.0
        )
        assert emitted.get((gas + ", non-fossil", urban), 0.0) == pytest.approx(
            total * (load.biogenic_carbon_kg + load.principal_carbon_kg[2]) / elements["C"]
        )
        for suffix, indices in (("fossil", (0,)), ("non-fossil", (1, 2))):
            group_carbon = (
                elements["C"] - load.biogenic_carbon_kg - load.principal_carbon_kg[2]
                if suffix == "fossil"
                else load.biogenic_carbon_kg + load.principal_carbon_kg[2]
            )
            rows = [
                row
                for row in result.amounts
                if isinstance(row.endpoint, PaReceivingFlow)
                and row.endpoint.name == gas + ", " + suffix
            ]
            if group_carbon == 0:
                assert rows == []
            elif any(load.principal_carbon_kg):
                assert len(rows) == 1
                assert rows[0].principal_carbon_origin_fractions == pytest.approx(
                    tuple(
                        part / group_carbon if index in indices else 0.0
                        for index, part in enumerate(load.principal_carbon_kg)
                    )
                )
            else:
                assert all(row.principal_carbon_origin_fractions is None for row in rows)
    assert terminal["Nitrate", surface] > 0
    if embedded:
        assert terminal["Phosphate", surface] > 0
    assert emitted["Nitrogen", ("air",)] > 0
    assert emitted["Nitrogen oxides", urban] > 0
    assert emitted["Dinitrogen monoxide", urban] > 0
    # Fixed thermal NOx belongs to combustion air. The portable waste gases
    # close actual waste N plus reacted ammonia before aqueous/solid shares.
    assert (
        emitted["Nitrogen", ("air",)]
        + emitted["Dinitrogen monoxide", urban] * 28 / 44
        + (emitted["Nitrogen oxides", urban] - 0.001 * mass) * 14 / 46
        + emitted["Cyanide", urban] * 14 / 26
    ) == pytest.approx(
        0.989 * elements["N"]
        + (supplies[database, "mc_ammonia"] - emitted["Ammonia", urban]) * 14 / 17
    )
    assert emitted["Particulate Matter, < 2.5 um", urban] > 0
    # Each retained native row keeps its own scaled distribution and identity.
    # Actual quantity/equations are checked above and at their owner;
    # this ratio tests only the native uncertainty transport boundary.
    operations = {
        operation.activity.key: operation
        for operation in (boundary.sources.hwi, boundary.sources.fixed, boundary.sources.ash)
    }
    for row in result.amounts:
        operation = operations[row.source.key]
        source_rows = [
            native for native in operation.rows if native.endpoint.key == row.endpoint.key
        ]
        remapped = row.endpoint.key == boundary.sources.pam.key
        if remapped:
            source_rows = [
                native
                for native in operation.rows
                if isinstance(native.endpoint, PaReceivingActivity)
                and native.endpoint.reference_product == "chemical, inorganic"
            ]
        if isinstance(row.endpoint, PaReceivingFlow) and row.endpoint.name in {
            "Carbon dioxide, non-fossil",
            "Carbon monoxide, non-fossil",
        }:
            remapped = True
            source_rows = [
                native
                for native in operation.rows
                if isinstance(native.endpoint, PaReceivingFlow)
                and native.endpoint.name == row.endpoint.name.replace("non-fossil", "fossil")
                and native.endpoint.categories == row.endpoint.categories
            ]
        if not source_rows:
            assert row.metadata == {"uncertainty type": 0, "loc": row.amount}
            continue
        assert len(source_rows) == 1
        native = source_rows[0]
        ratio = row.amount / native.amount
        for field in ("loc", "minimum", "maximum"):
            if field in native.metadata:
                expected = (
                    native.metadata[field] + log(ratio)
                    if field == "loc" and native.metadata.get("uncertainty type") == 2
                    else native.metadata[field] * ratio
                )
                assert row.metadata[field] == pytest.approx(expected, rel=2e-12, abs=1e-30)
        for field in ("scale", "shape", "uncertainty type", "negative", "allocation", "formula"):
            if field in native.metadata:
                assert row.metadata[field] == native.metadata[field]
        for field in ("activity", "flow"):
            if remapped:
                assert field not in row.metadata
            elif field in native.metadata:
                assert row.metadata[field] == native.metadata[field]
    assert {row.function for row in result.amounts} == {
        "pa_maintenance_incineration",
        "pa_maintenance_scr",
        "pa_maintenance_ash",
    }
    for row in result.amounts:
        if row.function == "pa_maintenance_incineration":
            assert row.source == boundary.sources.hwi.activity
        elif row.function == "pa_maintenance_ash":
            assert row.source == boundary.sources.ash.activity
        else:
            assert row.source in (boundary.sources.hwi.activity, boundary.sources.fixed.activity)
