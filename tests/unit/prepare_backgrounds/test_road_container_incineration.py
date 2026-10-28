"""Receive actual road-container minerals through the complete selected HWI."""

from math import fsum, isfinite, log
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from bw2calc import LCA
from chemical_residue_fixtures import (
    apply_premise_chemical_receiving_metadata,
    chemical_residue_biosphere,
    chemical_residue_native_inventory,
)
from material_recipe_fixtures import physical_material_recipe

from pytoce.download_data.sources.ecoinvent_materials import (
    ConversionRequirement,
    PrincipalExit,
)
from pytoce.prepare_backgrounds.preparation.chemical_residue_incineration import (
    chemical_residue_datasets,
)
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.coating_compilation import (
    _road_residue_load,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.road_material import _road_reference
from pytoce.shared.runtime.brightway import write_brightway_database


def _emissions(activity: Any, mass: float) -> dict[tuple[str, tuple[str, ...]], float]:
    """Sum stored native-unit elementary quantities per actual received batch."""
    result: dict[tuple[str, tuple[str, ...]], float] = {}
    for row in activity.biosphere():
        descriptor = str(row.input["name"]), tuple(row.input["categories"])
        result[descriptor] = result.get(descriptor, 0.0) + row["amount"] * mass
    return result


@pytest.mark.parametrize("original", [True, False])
def test_road_container_uses_known_hwi_minerals_and_no_incoming_haul(
    tmp_path: Path, original: bool
) -> None:
    """Exercise all origins with synthetic HWI support and explicit phase-factor expectations.

    Original ash composition remains the denominator when future properties
    disappear. Its dated positive mineral rows retain their own coefficients
    and uncertainty. Geological C is documented separately.
    """
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    database = "original" if original else "future"
    names = BrightwayNames(tmp_path / "brightway", "road_container", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(chemical_residue_biosphere(), searchable=False)
        for selected in ("original",) if original else ("original", "future"):
            data = chemical_residue_native_inventory(catalogue, selected)
            # The selected source includes outgoing ash freight, distinct from
            # the unused incoming collection market. Invent 0.025 t km/kg ash.
            data[selected, "mc_ash"]["exchanges"].append(
                {
                    "input": (selected, "cr_road"),
                    "type": "technosphere",
                    "amount": 0.025 * 5,
                    "uncertainty type": 0,
                }
            )
            if selected == "future":
                # A doubled functional unit must cancel; dated Al/K/Mg
                # coefficients independently rise 20% and must survive.
                for code in ("mc_hwi", "mc_fixed", "mc_ash"):
                    for row in data[selected, code]["exchanges"]:
                        factor = 2.0
                        if row["input"][1].startswith(
                            ("cr_Aluminium III_", "cr_Potassium I_", "cr_Magnesium_")
                        ):
                            factor *= 1.2
                        row["amount"] *= factor
                        if "loc" in row:
                            row["loc"] = (
                                row["loc"] + log(factor)
                                if row.get("uncertainty type") == 2
                                else row["loc"] * factor
                            )
                        for bound in ("minimum", "maximum"):
                            if bound in row:
                                row[bound] *= factor
            bd.Database(selected).write(data, searchable=False)
        records = {}
        if not original:
            apply_premise_chemical_receiving_metadata(bd, database)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario=None if original else "remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="receiving",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        # Source application: 0.223 kg principal, 1% homogeneous container
        # withdrawal and then 0.00202 kg MMA air loss leave 0.21875 kg principal.
        conversion = ConversionRequirement(
            operation="others_road_marking_conversion",
            principal_feed_kg_per_finished_kg=892 / 875,
            finished_kg_per_native_service_kg=1.0,
            principal_exits=(
                PrincipalExit("container_residue", 223 / 21875),
                PrincipalExit("methyl_methacrylate_air", 202 / 21875),
            ),
        )
        origins: tuple[Literal["fossil", "biogenic", "atmospheric_dac"], ...] = (
            "fossil",
            "biogenic",
            "atmospheric_dac",
        )
        materials = {
            origin: _road_reference(conversion, origin).container_residue for origin in origins
        }
        loads = {origin: _road_residue_load(material) for origin, material in materials.items()}
        for origin, load in loads.items():
            records.update(
                chemical_residue_datasets(
                    context,
                    load,
                    incineration_key=("receiving", origin + "_road_container"),
                    ash_key=("receiving", origin + "_road_ash"),
                    residue_name="road-marking container residue",
                    include_collection=False,
                )
            )
        write_brightway_database(bd, "receiving", records, {})
        for origin, load in loads.items():
            hkey, akey = (
                ("receiving", origin + "_road_container"),
                ("receiving", origin + "_road_ash"),
            )
            hwi, ash = (bd.get_node(database=k[0], code=k[1]) for k in (hkey, akey))
            reference = cast(dict[str, Any], records[hkey]["chemical_residue_reference"])
            mass, elements = load.wet_mass_kg, load.active_elements_kg
            material = materials[origin]
            assert mass == pytest.approx(0.0101 / 0.21875)
            assert load.unknown_mineral_kg == 0
            assert load.nonfuel_salt_elements_kg == {}
            assert elements["C"] == pytest.approx(
                fsum(material.principal_carbon_kg) + material.fossil_auxiliary_carbon_kg
            )
            assert reference["load"]["geological_carbon_kg"] == material.limestone_elements_kg["C"]
            assert load.geological_carbon_kg > 0
            inputs: dict[tuple[str, str], float] = {}
            for row in hwi.technosphere():
                inputs[row.input.key] = inputs.get(row.input.key, 0.0) + row["amount"] * mass
            assert (database, "cr_road") not in inputs
            assert (database, "cr_rail") not in inputs
            assert (database, "mc_hwi") not in inputs
            assert (database, "mc_fixed") not in inputs
            assert inputs[database, "mc_oil"] == pytest.approx(reference["oil_heat_mj"])
            assert inputs[database, "mc_oil"] > 0
            assert inputs[database, "mc_ng"] > 0
            assert inputs[database, "mc_pam"] == pytest.approx(reference["supplies"]["PAM"])
            assert inputs[database, "mc_tio2"] == pytest.approx(0.0002 * mass)
            assert reference["site_water_kg"] == 0
            ash_mass = -inputs[akey]
            # This is the source's oxide-equivalent service mass.
            assert ash_mass == pytest.approx(fsum(reference["ash"].values()))
            ash_inputs = {row.input.key: row["amount"] for row in ash.technosphere()}
            assert ash_inputs[database, "cr_road"] == pytest.approx(0.025)
            assert ash_inputs[database, "mc_cement_a"] + ash_inputs[
                database, "mc_cement_b"
            ] == pytest.approx(0.4)
            emitted, terminal = _emissions(hwi, mass), _emissions(ash, ash_mass)
            urban = ("air", "urban air close to ground")
            surface, long = ("water", "surface water"), ("water", "ground-, long-term")
            for element, name, air, solid in (
                ("Ti", "Titanium ion", 0.001, 0.999),
                ("Al", "Aluminium III", 0.00156, 0.998),
                ("K", "Potassium I", 0.00301, 0.997),
                ("Mg", "Magnesium", 0.00138, 0.999),
                ("Ca", "Calcium II", 0.00167, 0.998),
                ("Si", "Silicon", 0.00233, 0.998),
            ):
                assert emitted[name, urban] == pytest.approx(air * elements[element])
                fresh = reference["fresh"]["TiO2"]["Ti"] if element == "Ti" else 0.0
                if element == "Ca":
                    fresh += reference["fresh"]["CaO"]["Ca"]
                assert reference["ash"][element] == pytest.approx(solid * elements[element] + fresh)
            for element, name, donor, short, later in (
                ("Al", "Aluminium III", 0.07, 0.000031, 0.018),
                ("K", "Potassium I", 0.11, 0.023, 0.071),
                ("Mg", "Magnesium", 0.035, 0.000004, 0.0027),
            ):
                for phase, coefficient in ((surface, short), (long, later)):
                    expected = reference["ash"][element] / donor * coefficient
                    assert terminal[name, phase] == pytest.approx(
                        expected * (1 if original else 1.2)
                    )
                for row in ash.biosphere():
                    if row.input["name"] == name:
                        assert row["uncertainty type"] == 2
                        assert row["loc"] == pytest.approx(log(row["amount"]) + 0.02)
                        assert row["scale"] == 0.17
                        assert row["allocation"] == "invented chemical source"
            assert terminal["Titanium ion", surface] == pytest.approx(
                reference["ash"]["Ti"] * 0.0004956
            )
            assert terminal["Titanium ion", long] == pytest.approx(
                reference["ash"]["Ti"] * (0.2973 - 0.0004956)
            )
            # Track limestone carbon as inorganic carbon.
            assert not any(name == "Carbonate" for name, _ in (*emitted, *terminal))
            air_carbon = fsum(
                emitted.get((name + ", " + suffix, urban), 0.0) * carbon_fraction
                for name, carbon_fraction in (
                    ("Carbon dioxide", 12 / 44),
                    ("Carbon monoxide", 12 / 28),
                )
                for suffix in ("fossil", "non-fossil")
            )
            assert air_carbon == pytest.approx(0.9889898 * elements["C"])
            principal_air = [0.0, 0.0, 0.0]
            for row in hwi.biosphere():
                if row.input["name"].startswith(("Carbon dioxide,", "Carbon monoxide,")):
                    ratio = 12 / (44 if row.input["name"].startswith("Carbon dioxide,") else 28)
                    for index, fraction in enumerate(row["principal_carbon_origin_fractions"]):
                        principal_air[index] += row["amount"] * mass * ratio * fraction
            assert principal_air == pytest.approx(
                tuple(0.9889898 * c for c in load.principal_carbon_kg)
            )
            lca = LCA({hwi: -mass})
            lca.lci()
            assert all(isfinite(value) for value in lca.supply_array)
