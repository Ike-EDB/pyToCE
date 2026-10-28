"""Landfill preparation, LCI and carbon accounting use synthetic inventories."""

import copy
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from bw2calc import LCA
from provider_recipe_fixtures import _landfill_sources, _prepare_selected_native_frontier

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import catalogue as pc
from pytoce.run_lca.calculation.carbon import (
    carbon_consistency_table,
    direct_treatment_carbon_flows,
)


def _activation(
    database: str, service: str, origin: str, material: str, year: int
) -> dict[str, object]:
    """Activate one positive public treatment unit at a retained MFA coordinate."""
    return {
        "year": year,
        "region": "EUR",
        "life_cycle_stage": "landfill_treatment",
        "material": material,
        "constituent": material,
        "product": "polymer",
        "good": "all",
        "virgin_feedstock_route": "all",
        "material_cycle_source": "not_applicable",
        "ultimate_carbon_origin": origin,
        "lca_fate": "landfill",
        "fate_origin": "remind_mfa",
        "provider_database": database,
        "provider_code": pc.provider_service_code(service),
        "provider_anchor_year": year,
        "amount": 1.0,
        "mfa_carbon_amount_kg": 0.5,
    }


@pytest.mark.parametrize(
    ("year", "source_error"),
    [(2025, None), (2050, None), (2025, "missing_toc"), (2025, "zero_toc"), (2025, "normal_gas")],
)
def test_landfill_chain_preserves_full_lci_and_principal_carbon(
    tmp_path: Path, year: int, source_error: str | None
) -> None:
    """Preserve total gas and auxiliaries while assigning principal carbon exactly once."""
    catalogue = pc.load_provider_catalogue()
    chains = cast(dict[str, Any], catalogue["semantic_contract"])["landfill_carbon_origin"][
        "chains"
    ]
    selected = tuple(
        service for parent in chains for service in (parent, f"{parent}__non_fossil")
    ) + ("plastic_waste_landfill_row_non_fossil",)
    nodes, fractions = _landfill_sources(catalogue, dated=year == 2050)
    wastewater_rows = nodes[("native", "landfill_pe_wastewater_treatment")]["exchanges"]
    if source_error == "missing_toc":
        wastewater_rows[:] = [
            row for row in wastewater_rows if row["input"] != ("biosphere", "toc")
        ]
    elif source_error == "zero_toc":
        next(row for row in wastewater_rows if row["input"] == ("biosphere", "toc"))["amount"] = 0.0
    elif source_error == "normal_gas":
        next(row for row in wastewater_rows if row["input"] == ("biosphere", "co2_high_f"))[
            "uncertainty type"
        ] = 3
    biosphere = {
        ("biosphere", f"{species}_{compartment}_{suffix}"): {
            "name": f"{name}, {origin}",
            "unit": "kilogram",
            "categories": ("air", category),
            "type": "emission",
        }
        for species, name in (
            ("co2", "Carbon dioxide"),
            ("ch4", "Methane"),
            ("co", "Carbon monoxide"),
        )
        for compartment, category in (
            ("high", "non-urban air or from high stacks"),
            ("urban", "urban air close to ground"),
        )
        for suffix, origin in (("f", "fossil"), ("nf", "non-fossil"))
    }
    biosphere[("biosphere", "toc")] = {
        "name": "TOC, Total Organic Carbon",
        "unit": "kilogram",
        "categories": ("water", "surface water"),
        "type": "emission",
    }
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="landfill_origin",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere, searchable=False)
        bd.Database("native").write(nodes, searchable=False)
        source_before = {
            node.key: [copy.deepcopy(dict(exc)) for exc in node.exchanges()]
            for node in bd.Database("native")
        }
        if source_error is not None:
            expected_error = {
                "missing_toc": "Exactly one of each declared source input",
                "zero_toc": "Both source inputs must be positive",
                "normal_gas": "requires deterministic or lognormal gas rows",
            }[source_error]
            with pytest.raises(ValueError, match=expected_error):
                _prepare_selected_native_frontier(bd, catalogue, selected, year=year)
            return
        frontier = _prepare_selected_native_frontier(bd, catalogue, selected, year=year)
        db = frontier.database_name
        assert len(bd.Database(db)) == 57  # 14 native parents, one embedded parent, 42 children.
        activations = []
        for parent in chains:
            q, q_co2 = fractions[parent]
            for role in (
                "landfill",
                "wastewater_treatment",
                "sludge_incineration",
                "sludge_landfarming",
            ):
                service = (
                    f"{parent}__non_fossil"
                    if role == "landfill"
                    else f"{parent}__{role}__non_fossil"
                )
                node = bd.get_node(database=db, code=pc.provider_service_code(service))
                assert node.rp_exchange()["amount"] == (1.0 if role == "landfill" else -1.0)
                if role == "wastewater_treatment":
                    assert node["unit"] == "cubic meter"
                output = {
                    tuple(exc["input"]): dict(exc)
                    for exc in node.exchanges()
                    if exc["type"] == "biosphere"
                }
                for source in nodes[("native", f"{parent}_{role}")]["exchanges"]:
                    if source["type"] != "biosphere" or source["input"][1] == "toc":
                        continue
                    key = source["input"]
                    factor = (
                        1.0
                        if role == "landfill"
                        else q_co2
                        if role == "wastewater_treatment" and key[1].startswith("co2")
                        else q
                    )
                    principal = output[("biosphere", key[1][:-1] + "nf")]
                    assert principal["amount"] == pytest.approx(source["amount"] * factor)
                    assert principal["flow"] == key[1][:-1] + "nf"
                    if year == 2025:
                        assert (
                            principal["properties"]["carbon allocation"]
                            == source["properties"]["carbon allocation"]
                        )
                        assert (
                            principal["properties"]["carbon content"]
                            == source["properties"]["carbon content"]
                        )
                        assert principal["properties"]["carbon content, fossil"]["amount"] == 0.0
                        assert (
                            principal["properties"]["carbon content, non-fossil"]["amount"]
                            == source["properties"]["carbon content, fossil"]["amount"]
                        )
                    else:
                        assert "properties" not in principal
                    expected_loc = (
                        source["loc"] * factor if year == 2050 else source["loc"] + math.log(factor)
                    )
                    assert principal["loc"] == pytest.approx(expected_loc)
                    if year == 2025:
                        assert principal["scale"] == source["scale"]
                        assert (
                            principal["scale without pedigree"] == source["scale without pedigree"]
                        )
                    if factor < 1:
                        auxiliary = output[key]
                        assert principal["amount"] + auxiliary["amount"] == pytest.approx(
                            source["amount"]
                        )
                        assert auxiliary.get("properties") == source.get("properties")
                        assert (
                            auxiliary["pytoce_carbon_partition"]
                            == principal["pytoce_carbon_partition"]
                        )
                        assert auxiliary["loc"] == pytest.approx(
                            source["loc"] * (1 - factor)
                            if year == 2050
                            else source["loc"] + math.log(1 - factor)
                        )
            for origin, service in (
                ("fossil", parent),
                ("biogenic", f"{parent}__non_fossil"),
                ("atmospheric_dac", f"{parent}__non_fossil"),
            ):
                activations.append(_activation(db, service, origin, parent, year))
            fossil = bd.get_node(database=db, code=pc.provider_service_code(parent))
            non_fossil = bd.get_node(
                database=db, code=pc.provider_service_code(f"{parent}__non_fossil")
            )
            inventories = []
            for node in (fossil, non_fossil):
                lca = LCA({node: 1.0})
                lca.lci()
                inventory = lca.inventory.sum(axis=1).A1
                inventories.append(
                    {
                        key[1]: float(
                            inventory[
                                lca.dicts.biosphere[bd.get_node(database=key[0], code=key[1]).id]
                            ]
                        )
                        for key in biosphere
                    }
                )
            before, after = inventories
            for code in before:
                if code.endswith("_f"):
                    assert before[code] == pytest.approx(
                        after[code] + after[code[:-1] + "nf"], rel=2e-6
                    )
            assert before["toc"] == pytest.approx(after["toc"])
            assert (
                after["co2_high_f"] > 0.07
            )  # Native utility and flocculant burdens remain fossil.
        direct = direct_treatment_carbon_flows(bd, pd.DataFrame(activations))
        consistency = carbon_consistency_table(bd, pd.DataFrame(activations))
        for parent in chains:
            rows = direct.loc[direct["material"] == parent]
            totals = rows.groupby("mfa_carbon_origin")["carbon_amount"].sum()
            q, q_co2 = fractions[parent]
            source_gases = nodes[("native", f"{parent}_wastewater_treatment")]["exchanges"]
            wastewater = {
                species: sum(
                    row["amount"]
                    for row in source_gases
                    if row["type"] == "biosphere" and row["input"][1].startswith(species + "_")
                )
                for species in ("co2", "ch4")
            }
            # Independent signed duties: 1 parent, .08 WWT, .024 incineration, .048 landfarming.
            expected = (
                (0.2 + 0.08 * wastewater["co2"] * q_co2 + 0.024 * 0.05 * q + 0.048 * 0.04 * q)
                * 12
                / 44
                + (0.005 + 0.08 * wastewater["ch4"] * q + 0.024 * 0.0006 * q) * 12 / 16
                + (0.001 + 0.024 * 0.0007 * q) * 12 / 28
            )
            assert totals["fossil"] == pytest.approx(expected)
            assert totals["fossil"] == pytest.approx(totals["biogenic"])
            assert totals["fossil"] == pytest.approx(totals["atmospheric_dac"])
            assert len(rows.loc[rows["mfa_carbon_origin"] == "biogenic"]) == 11
            assert set(rows.loc[rows["mfa_carbon_origin"] == "biogenic", "flow_carbon_origin"]) == {
                "non_fossil"
            }
            assert cast(
                "pd.Series[float]",
                consistency.loc[
                    consistency["material"] == parent, "native_direct_selected_carbon_kg"
                ],
            ).tolist() == pytest.approx([totals["fossil"]] * 3)
        embedded = bd.get_node(
            database=db, code=pc.provider_service_code("plastic_waste_landfill_row_non_fossil")
        )
        assert embedded.rp_exchange()["amount"] == -1.0
        assert any(
            exc["input"]
            == (db, pc.provider_service_code("landfill_mixed_plastic__leachate_market__non_fossil"))
            for exc in embedded.exchanges()
        )
        # An embedded -1 kg service is consumed with a negative coefficient.
        # Its stored gas duties therefore have the opposite sign to the +1 public service.
        public = bd.get_node(
            database=db, code=pc.provider_service_code("landfill_mixed_plastic__non_fossil")
        )
        embedded_owners = json.loads(embedded["direct_treatment_carbon_owners_json"])
        public_owners = json.loads(public["direct_treatment_carbon_owners_json"])
        child_duties = {
            pc.provider_service_code(f"landfill_mixed_plastic__{role}__non_fossil"): amount
            for role, amount in (
                ("sludge_incineration", 0.024),
                ("sludge_landfarming", 0.048),
                ("wastewater_treatment", 0.08),
            )
        }
        assert {
            owner["owner_code"]: owner["owner_scale_per_provider_unit"] for owner in embedded_owners
        } == pytest.approx({embedded.key[1]: 1.0, **child_duties})
        assert {
            owner["owner_code"]: owner["owner_scale_per_provider_unit"] for owner in public_owners
        } == pytest.approx(
            {public.key[1]: 1.0, **{key: -value for key, value in child_duties.items()}}
        )
        assert {
            node.key: [dict(exc) for exc in node.exchanges()] for node in bd.Database("native")
        } == source_before
