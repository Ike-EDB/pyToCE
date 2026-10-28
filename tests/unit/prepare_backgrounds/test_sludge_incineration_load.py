"""Check TPS phosphorus partition and PA origin/calorific inputs for RoW treatment.

These calculations use synthetic original/future source inventories and the
package's formulation and wastewater calculations. One source variant lacks
phosphorus calibration and rows, representing the documented missing-P case.
Separate PA receiving and public workflow checks retain their own scope.
"""

from math import fsum
from pathlib import Path
from typing import cast

import pytest
from pa_landfill_receiving_fixtures import landfill_receiving_boundary
from test_tps_treatment import _loads
from test_tps_treatment import tps_boundary as tps_boundary
from test_tps_treatment import tps_catalogue as tps_catalogue

from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_load import (
    _pa_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    admit_pa_landfill_receiving_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration import (
    _sludge_source,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration_load import (
    SludgeIncinerationSource,
    calculate_sludge_incineration,
)


@pytest.mark.parametrize("origin", ("fossil", "biogenic"), ids=("actual_P", "known_zero_P"))
@pytest.mark.parametrize(
    "source_has_phosphorus", (False, True), ids=("absent_source_P", "native_P")
)
def test_report_phosphorus_phases_replace_source_once(
    tps_boundary, tps_catalogue, origin, source_has_phosphorus
):
    """Close actual elemental P through direct gases/water and both ash payloads."""
    _, context, _, _ = tps_boundary
    services, _, wastewater = _loads(tps_catalogue, origin)
    construction = services[f"tps_sludge_incineration_{origin}"]["construction"]
    admitted = _sludge_source(context, construction)
    # This original source property is omitted.
    # Keep every other admitted native input, output and intensive property.
    source = (
        admitted
        if source_has_phosphorus
        else SludgeIncinerationSource(
            {name: value for name, value in admitted.elements_kg.items() if name != "P"},
            admitted.inputs,
            {
                key: value
                for key, value in admitted.emissions.items()
                if key[0] not in {"Phosphorus", "Phosphate"}
            },
        )
    )
    load = calculate_sludge_incineration(
        source,
        wastewater,
        f"tps_{origin}",
        source_carbon_origins=tuple(dict.fromkeys(construction["carbon_templates"].values())),
        additional_apc_water_flows=tuple(construction["additional_apc_water_flows"]),
        phosphorus_phase_reference=True,
        # Selected plastic-leachate slag: 11.5628% water / 88.4372% dry.
        bottom_ash_water_to_dry_ratio=0.115628 / 0.884372,
    )
    p = wastewater.incineration_dry_share * wastewater.final_sludge.elements_kg["P"]
    assert (p > 0) is (origin == "fossil")
    air = ("Phosphorus", "kilogram", ("air", "urban air close to ground"))
    water = ("Phosphate", "kilogram", ("water", "surface water"))
    # Independent source cells: Wastewater2021 sludgeIncin L26/K26/G26 and
    # H26:J26. Municipal2020 water emissions B71 converts kg P to kg PO4.
    assert load.emissions[air] == pytest.approx(p * 0.00101624265046688, rel=2e-12, abs=0)
    assert load.emissions[water] == pytest.approx(
        p * 0.000110461157659443 * (31 + 4 * 16) / 31, rel=2e-12, abs=0
    )
    assert load.bottom_ash.waste_elements_kg["P"] == pytest.approx(
        p * 0.8536803006664344, rel=2e-12, abs=0
    )
    assert load.residual_ash.waste_elements_kg["P"] == pytest.approx(
        p * (0.0292960116233505 + 0.110153003703798 + 0.00574398019829103),
        rel=2e-12,
        abs=0,
    )
    assert fsum(
        (
            load.emissions[air],
            load.emissions[water] * 31 / (31 + 4 * 16),
            load.bottom_ash.waste_elements_kg["P"],
            load.residual_ash.waste_elements_kg["P"],
        )
    ) == pytest.approx(p, rel=2e-12, abs=0)
    assert [key for key in load.emissions if key[0] in {"Phosphorus", "Phosphate"}] == [air, water]
    assert source_has_phosphorus is ("P" in source.elements_kg)
    # The thermal/APC calculation returns output while preserving source rows and caller policy.
    assert load.thermal["delivered_reheat_MJ"] == load.duties["reheat"]
    assert load.bottom_ash.treatment_mass_kg > 0
    assert load.residual_ash.treatment_mass_kg > 0


@pytest.mark.parametrize("grade", ("pa6", "pa66"))
def test_actual_pa_sludge_preserves_origins_and_organic_calorific_input(
    tmp_path: Path, grade: str
) -> None:
    """Keep absent origins zero and exclude mass-estimation oxide O from PA heat."""
    # Accepted principal-only endpoint of the PA family, at the selected RoW
    # site's 2020 climate. The existing real source fixture supplies original
    # no-FLUWA operations; its quantities are invented test controls.
    finish = _pa_finish_reference(grade)
    landfill = _pa_landfill_load(
        finish,
        principal_by_origin_kg={"fossil": 1.0},
        finish_dose_kg=0.0,
        water_kg=0.04,
        site=_LandfillSite(16.743579766537, 1250.84108527132, 1.0),
    )
    with landfill_receiving_boundary(tmp_path, future=False) as (_, context):
        reference = admit_pa_landfill_receiving_reference(context)
        wastewater = _wastewater_load(
            landfill.leachate,
            volume_m3=0.0009,
            incoming_water_kg=0.9,
            evaporation_fraction=reference.evaporation_fraction,
            profile=_ROW_WASTEWATER_PROFILE,
        )
        service = next(
            row
            for row in cast(list[dict[str, object]], context.catalogue["services"])
            if row["provider_service_id"] == "sbr_filled_fossil_sludge_incineration"
        )
        load = calculate_sludge_incineration(
            _sludge_source(context, cast(dict[str, object], service["construction"])),
            wastewater,
            "pa_actual_sanitary_sludge",
            source_carbon_origins=("fossil",),
            additional_apc_water_flows=("Calcium II",),
            phosphorus_phase_reference=True,
            # Selected plastic-leachate slag: 11.5628% water / 88.4372% dry.
            bottom_ash_water_to_dry_ratio=0.115628 / 0.884372,
        )
    for species in ("Carbon dioxide", "Carbon monoxide", "Methane"):
        assert (
            load.emissions[
                species + ", non-fossil", "kilogram", ("air", "urban air close to ground")
            ]
            == 0.0
        )
        assert (
            load.emissions[species + ", fossil", "kilogram", ("air", "urban air close to ground")]
            > 0.0
        )
    assert fsum(pool.carbon_kg for pool in load.carbon if pool.origin != "fossil") == 0.0
    assert {pool.component for pool in load.carbon if pool.carbon_kg > 0} == {
        "principal_pa",
        "wastewater_polyacrylamide",
    }
    # The N-rich landfill leachate uses the specified nitrate loading and oxide-equivalent mass
    # descriptor separately.
    raw_organic_o = (
        0.628571428571428
        * (
            wastewater.raw_sludge.elements_kg["C"]
            - wastewater.duties.polyacrylamide_elements_kg["C"]
        )
        + wastewater.duties.polyacrylamide_elements_kg["O"]
    )
    organic_o = wastewater.incineration_dry_share * (
        (1 - 0.20525339975098886) * raw_organic_o
        + 0.628571428571428 * wastewater.digested_sludge.elements_kg["C"]
    )
    expected_hhv = (
        34.016 * load.elements_kg["C"]
        + 124.265 * load.allocated_hydrogen_kg
        - 9.8324 * organic_o
        + 6.276 * load.elements_kg["N"]
        + 19.079 * load.elements_kg["S"]
    )
    assert organic_o < load.allocated_oxygen_kg
    assert load.thermal["calorific_organic_oxygen_kg"] == pytest.approx(organic_o)
    assert load.thermal["HHV_MJ"] == pytest.approx(expected_hhv, rel=2e-12, abs=0)
    assert load.thermal["HHV_MJ"] > 0
    assert load.thermal["dissipated_heat_MJ"] > 0
    assert load.thermal["source_LHV_MJ"] == pytest.approx(
        expected_hhv - 2.2 * load.water_kg - 2.2 * 9 * load.allocated_hydrogen_kg,
        rel=2e-12,
        abs=0,
    )
