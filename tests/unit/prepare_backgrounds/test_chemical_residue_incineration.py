"""Protect chemical-residue fates and complete source supply through Brightway."""

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from chemical_residue_fixtures import chemical_residue_biosphere, chemical_residue_native_inventory
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.chemical_residue_incineration import (
    chemical_residue_datasets,
)
from pytoce.prepare_backgrounds.preparation.chemical_residue_load import ChemicalResidueLoad
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.shared.runtime.brightway import write_brightway_database


def _elementary_amounts(activity: Any, scale: float) -> dict[tuple[str, tuple[str, ...]], float]:
    """Sum actual stored elementary exchanges on the caller's receiving basis."""
    quantities: dict[tuple[str, tuple[str, ...]], float] = {}
    for exchange in activity.biosphere():
        key = str(exchange.input["name"]), tuple(exchange.input["categories"])
        quantities[key] = quantities.get(key, 0.0) + float(exchange["amount"]) * scale
    return quantities


@pytest.mark.parametrize(
    "salts,energy,unknown_mineral,biogenic_carbon",
    [
        (False, -0.2, 0.0, 0.0),
        (True, -0.2, 0.0, 0.0),
        (True, 30.0, 0.0, 0.0),
        (True, -0.2, 0.08, 0.04),
        (False, -0.2, 0.0, 0.1),
    ],
)
def test_actual_wet_pool_keeps_water_fuel_catalyst_and_terminal_once(
    tmp_path: Path, salts: bool, energy: float, unknown_mineral: float, biogenic_carbon: float
) -> None:
    """Solve native supply with known salts, metals and qualified mineral functions.

    The unknown-mineral case retains its full noncarbon environmental function
    and dry terminal mass. Known Ni/Si remain separately specified input
    quantities in this invented witness. Pure/mixed carbon origins, the two
    energy regimes and source-convention versus CIAAW salt bodies are explicit.
    """
    provider = prepared_provider_authority()
    elements = {
        "C": 0.1,
        "H": 0.02,
        "O": 0.03,
        "N": 0.005,
        "S": 0.01 if salts else 0.0,
        "Na": 0.015 if salts else 0.0,
        "Ni": 0.003 if unknown_mineral else 0.0,
        "Si": 0.007 if unknown_mineral else 0.0,
    }
    sulfate = 0.052 if salts else 0.0
    # The surviving PA diglycerol residue retains intact K2SO4, independently
    # of active waste sulfur and the fresh treatment reagents.
    salt_molar_mass = 2 * 39.0983 + 32.06 + 4 * 15.999
    nonfuel = (
        {
            "Potassium sulfate": {
                "K": sulfate * 2 * 39.0983 / salt_molar_mass,
                "S": sulfate * 32.06 / salt_molar_mass,
                "O": sulfate * 4 * 15.999 / salt_molar_mass,
            }
        }
        if salts
        else {}
    )
    water = 0.6
    mass = math.fsum(elements.values()) + water + sulfate + unknown_mineral
    load = ChemicalResidueLoad(
        mass,
        water,
        elements,
        energy,
        nonfuel,
        biogenic_carbon,
        unknown_mineral,
        waste_formed_water_kg=9 * elements["H"],
    )
    names = BrightwayNames(tmp_path / "brightway", "chemical_residue", "biosphere", "source")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(chemical_residue_biosphere(), searchable=False)
        bd.Database("source").write(
            chemical_residue_native_inventory(provider.catalogue, "source"), searchable=False
        )
        native_resolver = _ActivityResolver(bd, "source")
        context = _RecipeContext(
            background_scenario=None,
            year=2025,
            provider_database="receiving",
            source=native_resolver,
            ordinary=native_resolver,
            technology=native_resolver,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=provider.catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(provider.catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        records = {}
        hwi_key, ash_key = ("receiving", "actual_hwi"), ("receiving", "actual_ash")
        actual = chemical_residue_datasets(
            context,
            load,
            incineration_key=hwi_key,
            ash_key=ash_key,
            residue_name="invented chemical residue",
        )
        records.update(actual)
        write_brightway_database(bd, "receiving", records, {})
        hwi, ash = (bd.get_node(database=k[0], code=k[1]) for k in (hwi_key, ash_key))
        assert next(iter(hwi.production()))["amount"] == -1
        assert next(iter(ash.production()))["amount"] == -1
        q = cast(dict[str, Any], actual[hwi_key]["chemical_residue_reference"])
        deficit = max(23 * mass - energy, 0)
        oil_heat = deficit / (0.03125 * 42.6)
        assert q["oil_heat_mj"] == pytest.approx(oil_heat)
        assert q["oil_kg"] * 42.6 == pytest.approx(deficit)
        assert q["net_electricity_mj"] == pytest.approx(
            0.1 * (energy + deficit) - 0.287 * 3.6 * mass
        )
        inputs = {}
        for ex in hwi.technosphere():
            inputs[ex.input.key] = inputs.get(ex.input.key, 0.0) + ex["amount"] * mass
        assert inputs.get(("source", "mc_oil"), 0.0) == pytest.approx(oil_heat)
        assert inputs[("source", "cr_road")] == pytest.approx(0.04 * mass)
        assert inputs[("source", "cr_rail")] == pytest.approx(0.007 * mass)
        assert ("source", "mc_hwi") not in inputs
        assert ("source", "mc_fixed") not in inputs
        # Fixed wet-throughput and waste-N contributions share their original
        # supplier nodes, so their two demands must be summed exactly once.
        nitrogen_ratio = 0.005 / 0.01 + unknown_mineral
        assert inputs["source", "mc_tio2"] == pytest.approx(0.0002 * mass + 0.0001 * nitrogen_ratio)
        assert inputs["source", "mc_v_proxy"] == pytest.approx(
            0.00005 * mass + 0.00002 * nitrogen_ratio
        )
        assert inputs[("source", "mc_pam")] == pytest.approx(q["supplies"]["PAM"])
        assert ("source", "mc_inorganic") not in inputs
        # The complete TMT supplier retains its native active-mass reference.
        assert inputs.get(("source", "mc_organic"), 0.0) == pytest.approx(0.0001 * unknown_mineral)
        assert inputs.get(("source", "cr_calcium_chloride"), 0.0) == pytest.approx(
            0.008 * unknown_mineral
        )
        # Native metal pool in the fixture, ppm, with Doka non-air fractions.
        captured_native_metal = sum(
            ppm / 1e6 * capture
            for ppm, capture in (
                (10, 1 - 0.00899),
                (20, 1 - 0.0007),
                (30, 1 - 0.0000000739),
                (40, 1 - 0.0007),
                (5, 1 - 0.0402),
                (20, 1 - 0.0007),
                (50, 1 - 0.00331),
                (100, 1 - 0.0007),
            )
        )
        ferric = 0.006 * (elements["Ni"] * (1 - 0.0007) / captured_native_metal + unknown_mineral)
        assert inputs.get(("source", "cr_ferric"), 0.0) == pytest.approx(ferric)
        water_resource = (
            sum(
                ex["amount"]
                for ex in hwi.biosphere()
                if ex.input["name"] == "Water, well, in ground"
            )
            * mass
            * 1000
        )
        # The native fixed meter supplies 5 kg water/kg wet waste. Actual
        # incoming moisture and water formed from waste/fresh NH3 leave once.
        assert water_resource == pytest.approx(5 * mass)
        reacted_ammonia = (0.002 - 0.00002) * mass + (0.0001 - 0.000007) * nitrogen_ratio
        formed_water = 9 * elements["H"] + reacted_ammonia * 27 / 17
        assert q["formed_water_kg"] == pytest.approx(formed_water)
        water_output = (
            sum(ex["amount"] for ex in hwi.biosphere() if ex.input["name"] == "Water") * mass * 1000
        )
        assert water_output == pytest.approx(water + 5 * mass + formed_water)
        ash_inputs = {ex.input.key: ex["amount"] for ex in ash.technosphere()}
        assert ash_inputs[("source", "mc_cement_a")] + ash_inputs[
            ("source", "mc_cement_b")
        ] == pytest.approx(0.4)
        assert ash_inputs[("source", "mc_tap_a")] == pytest.approx(0.6)
        assert ash_inputs[("source", "mc_hydrated_cement")] == -1
        ash_m = -inputs[ash_key]
        emitted = _elementary_amounts(hwi, mass)
        terminal = _elementary_amounts(ash, ash_m)
        surface, long = ("water", "surface water"), ("water", "ground-, long-term")
        urban = ("air", "urban air close to ground")
        tmt_mass = 3 * (22.99 + 12 + 14 + 32)
        tmt = 0.0001 * unknown_mineral
        fresh_nitrogen = q["supplies"]["PAM"] * 14 / 71 + tmt * 3 * 14 / tmt_mass
        # Each invented source row retains its own period coefficient and
        # native 0.03 kg N/kg ash denominator; no full-release normalization.
        for phase, coefficient in ((surface, 0.006), (long, 0.012)):
            assert terminal["Nitrate", phase] == pytest.approx(fresh_nitrogen * coefficient / 0.03)
        intact_sulfate = sulfate * (32.06 + 4 * 15.999) / salt_molar_mass
        salt_potassium = sulfate * 2 * 39.0983 / salt_molar_mass
        assert emitted.get(("Potassium I", urban), 0.0) == pytest.approx(0.00301 * salt_potassium)
        assert emitted.get(("Sulfate", urban), 0.0) == pytest.approx(0.00301 * intact_sulfate)
        for phase, fraction, coefficient in ((surface, 0.2819, 0.003), (long, 0.7181, 0.02)):
            assert terminal.get(("Potassium I", phase), 0.0) == pytest.approx(
                salt_potassium * (1 - 0.00301) * fraction
            )
            assert terminal.get(("Sulfate", phase), 0.0) == pytest.approx(
                tmt * 3 * 32 / tmt_mass * coefficient / 0.012
                + intact_sulfate * (1 - 0.00301) * fraction
            )
        carbon = sum(
            emitted.get((name, urban), 0.0) * ratio
            for name, ratio in (
                ("Carbon dioxide, fossil", 12 / 44),
                ("Carbon dioxide, non-fossil", 12 / 44),
                ("Carbon monoxide, fossil", 12 / 28),
                ("Carbon monoxide, non-fossil", 12 / 28),
                ("Cyanide", 12 / 26),
            )
        )
        assert carbon == pytest.approx(0.9889898 * elements["C"])
        bio_share = biogenic_carbon / elements["C"]
        for gas in ("Carbon dioxide", "Carbon monoxide"):
            fossil = emitted.get((gas + ", fossil", urban), 0.0)
            bio = emitted.get((gas + ", non-fossil", urban), 0.0)
            assert bio == pytest.approx((fossil + bio) * bio_share)
            for exchange in hwi.biosphere():
                if exchange.input["name"].startswith(gas + ","):
                    assert exchange["loc"] == pytest.approx(math.log(exchange["amount"]) + 0.03)
                    assert exchange["scale"] == 0.21
                    assert exchange["allocation"] == "invented source allocation"
        # Known metal/Si pools exclude the unassayed mineral function. Its
        # existing native emissions remain additional qualified noncarbon duties.
        assert emitted.get(("Nickel II", urban), 0.0) == pytest.approx(
            0.0007 * elements["Ni"] + 0.000001 * unknown_mineral
        )
        assert emitted.get(("Nickel II", surface), 0.0) == pytest.approx(
            0.205 * elements["Ni"] + 0.000004 * unknown_mineral
        )
        assert emitted.get(("Nickel II", long), 0.0) == pytest.approx(0.000009 * unknown_mineral)
        assert q["ash"]["Ni"] == pytest.approx(0.795 * elements["Ni"])
        assert terminal.get(("Nickel II", surface), 0.0) == pytest.approx(
            q["ash"]["Ni"] * 0.002 / 0.02
        )
        assert terminal.get(("Nickel II", long), 0.0) == pytest.approx(
            q["ash"]["Ni"] * 0.005 / 0.02
        )
        assert emitted.get(("Silicon", urban), 0.0) == pytest.approx(
            0.00233 * elements["Si"] + 0.000002 * unknown_mineral
        )
        assert terminal.get(("Silicon", surface), 0.0) == pytest.approx(
            0.998 * elements["Si"] * 0.006 / 0.03
        )
        assert terminal.get(("Silicon", long), 0.0) == pytest.approx(
            0.998 * elements["Si"] * 0.03 / 0.03
        )
        assert emitted.get(("Zinc II", surface), 0.0) == pytest.approx(0.00001 * unknown_mineral)
        assert emitted.get(("Silicon", surface), 0.0) == pytest.approx(0.00003 * unknown_mineral)
        assert emitted.get(("Silicon", long), 0.0) == pytest.approx(0.00008 * unknown_mineral)
        # Fresh active reagents retain their elemental destinations. TMT organic atoms use the
        # auxiliary-carbon account.
        assert emitted[("Sodium I", surface)] == pytest.approx(
            q["supplies"]["NaOH"] * 22.99 / (22.99 + 17) + tmt * 3 * 22.99 / tmt_mass
        )
        assert emitted[("Chloride", surface)] == pytest.approx(
            q["supplies"]["HCl"] * 35.45 / 36.45
            + ferric * 3 * 35.45 / (55.845 + 3 * 35.45)
            + 0.008 * unknown_mineral * 2 * 35.45 / (40.078 + 2 * 35.45)
        )
        assert terminal.get(("Iron ion", surface), 0.0) == pytest.approx(
            ferric * 55.845 / (55.845 + 3 * 35.45) * 0.001 / 0.04
        )
        assert terminal.get(("Iron ion", long), 0.0) == pytest.approx(
            ferric * 55.845 / (55.845 + 3 * 35.45) * 0.003 / 0.04
        )
        expected_known_ash = (
            0.0110001 * elements["C"]
            + 0.01 * elements["N"] * 62 / 14
            + 0.0428 * elements["S"] * 80 / 32
            + 0.991 * elements["Na"] * (1 + 16 / (2 * 22.99))
            + sum(q["supplies"][name] for name in ("PAM", "CaO", "TiO2", "V2O5"))
            + sulfate * (1 - 0.00301)
            + 0.795 * elements["Ni"] * (1 + 15.999 / 58.6934)
            + 0.998 * elements["Si"] * (1 + 2 * 15.999 / 28.085)
            + ferric * (55.845 + 24) / (55.845 + 3 * 35.45)
            + 0.008 * unknown_mineral * (40.078 + 16) / (40.078 + 2 * 35.45)
            + tmt * 3 * (12 + 14 + 32) / tmt_mass
        )
        # The terminal receives the complete unassayed dry-mineral mass.
        assert ash_m - expected_known_ash == pytest.approx(unknown_mineral, abs=1e-14)
        # Oxygen metadata describes the oxide-equivalent material. Intact salts and unassayed
        # minerals retain their whole masses.
        assert math.fsum(q["ash"].values()) + sulfate * (
            1 - 0.00301
        ) + unknown_mineral == pytest.approx(ash_m, abs=1e-14)
        fresh_carbon = q["supplies"]["PAM"] * 36 / 71 + tmt * 36 / tmt_mass
        placed_carbon = terminal[
            ("Organic carbon, placed in landfill", ("inventory indicator", "waste"))
        ]
        assert carbon + emitted[
            ("DOC, Dissolved Organic Carbon", surface)
        ] + placed_carbon - fresh_carbon == pytest.approx(elements["C"])
        assert terminal[("DOC, Dissolved Organic Carbon", surface)] == pytest.approx(
            0.1 * fresh_carbon + (0.00003 / 0.3 - 0.0000101) * elements["C"]
        )
        assert terminal[("DOC, Dissolved Organic Carbon", long)] == pytest.approx(
            0.4 * fresh_carbon + 0.002 / 0.3 * elements["C"]
        )
        assert terminal.get(("Sodium I", surface), 0.0) == pytest.approx(
            0.991 * elements["Na"] * 0.02 / 0.1
        )
        assert terminal.get(("Sodium I", long), 0.0) == pytest.approx(
            0.991 * elements["Na"] * 0.07 / 0.1
        )
        pam = next(ex for ex in hwi.technosphere() if ex.input.key == ("source", "mc_pam"))
        assert pam["scale"] == 0.21 and pam["allocation"] == "invented source allocation"
        assert pam["loc"] == pytest.approx(math.log(pam["amount"]) + 0.03)
        # Changed source quantities belong to original/generated trace pairs;
        # "retained" means the native row itself survives unchanged.
        traces = {
            r["activity_code"]: r
            for r in context.technology_rows
            if r["provider_service_id"] in {hwi_key[1], ash_key[1]}
        }
        changed_by_source: dict[str, dict[str, Any]] = {}
        for code, target in (("mc_hwi", hwi), ("mc_fixed", hwi), ("mc_ash", ash)):
            trace = traces[code]
            assert json.loads(cast(str, trace["retained_technosphere_json"])) == []
            assert json.loads(cast(str, trace["retained_biosphere_json"])) == []
            terms = json.loads(cast(str, trace["project_terms_json"]))
            # JSON stores ordered carbon tuples as arrays. Compare the complete
            # serialized values, retaining every quantity and provenance field.
            assert {
                k: v for k, v in terms.items() if k != "scaled_or_replaced_native_rows"
            } == json.loads(json.dumps(q))
            pairs = terms["scaled_or_replaced_native_rows"]
            changed_by_source[code] = {}
            for pair in pairs:
                original = json.loads(pair["source_rows_json"])[0]
                final = json.loads(pair["generated_rows_json"])[0]
                removed = json.loads(cast(str, trace[f"removed_{original['exchange_type']}_json"]))
                assert original in removed and original != final
                assert any(
                    ex.input.key == (final["database_name"], final["code"])
                    and ex["amount"] == final["amount"]
                    for ex in target.exchanges()
                )
                changed_by_source[code][original["code"]] = final
        assert changed_by_source["mc_hwi"]["mc_inorganic"]["code"] == "mc_pam"
        assert changed_by_source["mc_hwi"]["mc_hcl"]["amount"] == pytest.approx(
            q["supplies"]["HCl"] / mass
        )
        # The intact fixed well duty is normalized from the fixture's four-kg
        # source reference (0.020 m3) to one kg (0.005 m3), so its trace pair
        # remains scaled rather than an unchanged native row.
        assert changed_by_source["mc_fixed"]["mc_water_resource"]["amount"] == pytest.approx(0.005)
        assert changed_by_source["mc_ash"]["mc_nitrate_short"]["amount"] == pytest.approx(
            (q["supplies"]["PAM"] * 14 / 71 + tmt * 3 * 14 / tmt_mass) * 0.006 / 0.03 / ash_m
        )
        collection = traces["cr_collection"]
        assert {
            r["code"]: r["amount"]
            for r in json.loads(cast(str, collection["retained_technosphere_json"]))
        } == {"cr_road": 0.04, "cr_rail": 0.007}
        assert (
            json.loads(cast(str, collection["project_terms_json"]))[
                "scaled_or_replaced_native_rows"
            ]
            == []
        )
        assert [
            r["code"] for r in json.loads(cast(str, collection["removed_technosphere_json"]))
        ] == ["mc_hwi"]
        lca = LCA({hwi: -mass})
        lca.lci()
        supply = cast(Any, lca.supply_array)
        assert all(math.isfinite(value) for value in supply)
        assert supply[lca.dicts.activity[ash.id]] == pytest.approx(ash_m)
        for code in ("mc_tio2", "mc_v_proxy", "mc_pam"):
            node = bd.get_node(database="source", code=code)
            assert supply[lca.dicts.activity[node.id]] == pytest.approx(inputs[node.key], rel=2e-7)
