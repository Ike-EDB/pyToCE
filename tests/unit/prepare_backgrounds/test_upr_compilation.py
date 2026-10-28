"""Actual UPR conversion through complete dated native supply and receiving."""

from copy import deepcopy
from dataclasses import replace
from math import fsum, log
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _process

from pytoce.download_data.sources.ecoinvent_materials import (
    ConversionRequirement,
    PrincipalExit,
)
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)


@pytest.mark.parametrize(("year", "ordinary", "factor"), [(2025, "native", 1), (2050, "dated", 2)])
def test_upr_frontier_retains_native_operation_and_actual_formulation_once(
    tmp_path: Path,
    year: int,
    ordinary: str,
    factor: int,
) -> None:
    """Solve all origins with actual normalization and nonunit native LCI.

    Curative and terminal functions are bounded at complete native supplied
    operations through real project recipes. The actual trim-market adapter is
    compiled here; terminal fate equations are tested at their own owners. The
    FR fixture retains its full 23-row functional shape and disposal market.
    """
    catalogue = load_provider_catalogue()
    catalogue_services = cast(list[dict[str, Any]], catalogue["services"])
    services = {row["provider_service_id"]: row for row in catalogue_services}
    conversions = [
        row for row in catalogue_services if row["construction"]["kind"] == "upr_conversion"
    ]
    assert {row["construction"]["principal_carbon_origin"] for row in conversions} == {
        "fossil",
        "biogenic",
        "atmospheric_dac",
    }
    construction = conversions[0]["construction"]
    residues = [
        services[row["construction"]["actual_residue"]["service_id"]] for row in conversions
    ]
    selected = [*conversions, *residues]
    dependency_selectors = {}
    for row in conversions:
        for selector in row["construction"]["supplies"].values():
            if "service_id" in selector:
                identifier = selector["service_id"]
                dependency_selectors[identifier] = {
                    "name": "complete native " + identifier,
                    "reference_product": services[identifier]["reference_product"],
                    "unit": "kilogram",
                    "exact_location": "RER",
                }
    terminal_roles = {}
    for row in residues:
        for role, selector in row["construction"]["treatments"].items():
            identifier = selector["service_id"]
            terminal_roles[identifier] = role
            dependency_selectors[identifier] = {
                "name": "complete native " + identifier,
                "reference_product": services[identifier]["reference_product"],
                "unit": "kilogram",
                "exact_location": services[identifier]["location"],
            }
    for identifier, selector in dependency_selectors.items():
        selected.append(
            {
                **services[identifier],
                "construction": {
                    "kind": "project_recipe",
                    "components": [
                        {
                            "component_id": "complete_native_supply",
                            "source_role": "native_ecoinvent_provider",
                            "selector": selector,
                            "amount": 1.0,
                        }
                    ],
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity="bounded_upr_application",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    # All 19 native technosphere roles survive, including tooling resin/glass,
    # signed cut-off scrap, solvent receiving and sanitation wastewater.
    operation_inputs = (
        ("acetone", "acetone, liquid", 0.031, "kilogram"),
        ("abs", "acrylonitrile-butadiene-styrene copolymer", 0.00024, "kilogram"),
        ("electricity", "electricity, low voltage", 0.48, "kilowatt hour"),
        ("cotton", "fibre, cotton", 0.00017, "kilogram"),
        ("polyester_fibre", "fibre, polyester", 0.0007, "kilogram"),
        ("tool_glass", "glass fibre", 0.0013, "kilogram"),
        ("heat", "heat, district or industrial, natural gas", 3.08, "megajoule"),
        ("scrap", "iron scrap, unsorted", -0.0164, "kilogram"),
        ("factory", "packaging box factory", 2e-10, "unit"),
        ("mould_resin", "polyester resin, unsaturated", 0.00303, "kilogram"),
        ("pet", "polyethylene terephthalate, granulate, amorphous", 0.019, "kilogram"),
        ("pe", "polyethylene, high density, granulate", 0.019, "kilogram"),
        ("pp", "polypropylene, granulate", 0.019, "kilogram"),
        ("spent_solvent", "spent solvent mixture", -0.005, "kilogram"),
        ("steel", "steel, low-alloyed", 0.0164, "kilogram"),
        ("rubber", "synthetic rubber", 0.00001, "kilogram"),
        ("textile", "textile, nonwoven polypropylene", 0.000352, "kilogram"),
        ("mixed_waste", "waste plastic, mixture", -0.186, "kilogram"),
        ("wastewater", "wastewater, average", -0.0007, "cubic meter"),
    )
    supply_selectors = {
        label: dependency_selectors[selector["service_id"]]
        if "service_id" in selector
        else selector
        for label, selector in construction["supplies"].items()
    }
    # Retain every FR-shaped fixture share at its original value, including nonzero open burning.
    treatment_shares = {
        "open_burning": 0.00417186912852553,
        "sanitary_landfill": 0.380706302185215,
        "municipal_incineration": 0.61512182868626,
    }
    markers = {
        "freight",
        "residue_freight",
        *("supply_" + label for label in supply_selectors),
        *("terminal_" + role for role in treatment_shares),
        *("donor_" + role for role in treatment_shares),
        *(
            "operation_" + label
            for label, _, _, _ in operation_inputs
            if label not in {"scrap", "mixed_waste"}
        ),
    }

    def inventory(database: str, multiplier: int) -> dict[tuple[str, str], dict[str, Any]]:
        """Complete native markets and signed receiver leaves expose solved demands."""
        data = {}

        def exchange(code: str, amount: float, *, bio: bool = False) -> dict[str, Any]:
            """Give native amounts signed distributions and independent allocation metadata."""
            return {
                "input": ("biosphere" if bio else database, code),
                "type": "biosphere" if bio else "technosphere",
                "amount": amount,
                "uncertainty type": 2,
                "loc": log(abs(amount)) + 0.01,
                "scale": 0.17,
                "minimum": min(amount / 2, amount * 2),
                "maximum": max(amount / 2, amount * 2),
                "negative": amount < 0,
                "allocation": "native UPR fixture allocation",
            }

        def leaf(
            code: str,
            name: str,
            product: str,
            unit: str,
            marker: str,
            location: str = "RER",
            sign: int = 1,
        ) -> None:
            """Supply one external function with nonunit output and observable inventory."""
            data[database, code] = {
                **_process(
                    database,
                    code,
                    name,
                    product,
                    location=location,
                    production_amount=3 * sign,
                    exchanges=[exchange(marker, 6 * multiplier, bio=True)] if marker else [],
                ),
                "unit": unit,
            }

        leaf("freight", "complete freight", "freight", "ton kilometer", "freight")
        for label, selector in supply_selectors.items():
            product = selector["reference_product"]
            leaf("producer_" + label, "producer " + label, product, "kilogram", "supply_" + label)
            basis = 4 if multiplier == 1 else 7
            location = (
                selector.get("exact_location")
                or selector["location_by_period"]["baseline" if multiplier == 1 else "future"]
            )
            data[database, "supply_" + label] = _process(
                database,
                "supply_" + label,
                selector["name"],
                product,
                location=location,
                production_amount=basis,
                exchanges=[
                    exchange("producer_" + label, 1.02 * basis),
                    exchange("freight", 0.011 * basis),
                    exchange("freight", 0.017 * basis),
                ],
            )
        for identifier, role in terminal_roles.items():
            selector = dependency_selectors[identifier]
            leaf(
                identifier,
                selector["name"],
                selector["reference_product"],
                "kilogram",
                "terminal_" + role,
                location=selector["exact_location"],
            )
        leaf(
            "residue_freight",
            "complete waste collection freight",
            "waste freight",
            "ton kilometer",
            "residue_freight",
        )
        market_basis = 3 if multiplier == 1 else 7
        market_rows = [exchange("residue_freight", 0.075154404308835 * market_basis * multiplier)]
        for role, share in treatment_shares.items():
            leaf(
                "donor_" + role,
                "treatment of waste plastic, mixture, " + role.replace("_", " "),
                "waste plastic, mixture",
                "kilogram",
                "donor_" + role,
                sign=-1,
            )
            amount = -share * market_basis
            row = exchange("donor_" + role, amount)
            if multiplier == 1:
                row["uncertainty type"] = 0
                row["loc"] = amount
                if role == "open_burning":
                    # A deterministic exchange retains its defined fields when waste demand is
                    # converted.
                    for field in ("negative", "minimum", "maximum"):
                        del row[field]
            row.update(
                {
                    "name": "waste plastic, mixture",
                    "unit": "kilogram",
                    "activity": "native waste producer",
                    "flow": "native mixed waste",
                    "properties": {"dry mass": {"amount": 0.8469, "unit": "kg"}},
                    "classifications": {"CPC": "39270"},
                }
            )
            market_rows.append(row)
        market = residues[0]["construction"]["source"]
        data[database, "residue_market"] = _process(
            database,
            "residue_market",
            market["name"],
            market["reference_product"],
            location=market["exact_location"],
            production_amount=-market_basis,
            exchanges=market_rows,
        )
        basis = 2 if multiplier == 1 else 5
        rows = []
        for label, product, amount, unit in operation_inputs:
            if label == "mixed_waste":
                rows.append(exchange("residue_market", amount * basis * multiplier))
                continue
            leaf(
                "operation_" + label,
                "native FR " + label,
                product,
                unit,
                "operation_" + label if label != "scrap" else "",
                sign=-1 if amount < 0 and label != "scrap" else 1,
            )
            rows.append(exchange("operation_" + label, amount * basis * multiplier))
        rows.extend(
            exchange(code, amount * basis * multiplier, bio=True)
            for code, amount in (
                ("acetone_air", 0.026),
                ("styrene_air", 0.032),
                ("sanitation_water", 0.0007),
            )
        )
        operation = construction["operation_source"]
        data[database, "operation"] = _process(
            database,
            "operation",
            operation["name"],
            operation["reference_product"],
            location=operation["exact_location"],
            production_amount=basis,
            exchanges=rows,
        )
        return data

    names = BrightwayNames(tmp_path / "brightway", "upr_application", "biosphere", "native")
    with selected_brightway_project(names) as bd:
        biosphere = {
            ("biosphere", marker): {
                "name": "UPR marker " + marker,
                "unit": "kilogram",
                "categories": ("air",),
                "type": "emission",
            }
            for marker in markers
        }
        for code, name, unit, categories, kind in (
            (
                "acetone_air",
                "Acetone",
                "kilogram",
                ("air", "urban air close to ground"),
                "emission",
            ),
            (
                "styrene_air",
                "Styrene",
                "kilogram",
                ("air", "urban air close to ground"),
                "emission",
            ),
            (
                "nmvoc_air",
                "NMVOC, non-methane volatile organic compounds",
                "kilogram",
                ("air", "urban air close to ground"),
                "emission",
            ),
            ("water_air", "Water", "cubic meter", ("air", "urban air close to ground"), "emission"),
            (
                "sanitation_water",
                "Water, unspecified natural origin",
                "cubic meter",
                ("natural resource", "in water"),
                "natural resource",
            ),
        ):
            biosphere["biosphere", code] = {
                "name": name,
                "unit": unit,
                "categories": categories,
                "type": kind,
            }
        bd.Database("biosphere").write(biosphere, searchable=False)
        for database, multiplier in (("native", 1), ("dated", 2)):
            bd.Database(database).write(inventory(database, multiplier), searchable=False)
        before = {
            node.key: (deepcopy(dict(node)), [dict(row) for row in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
        # Use the retained reference. The parser fixture uses separate
        # synthetic UPR coefficients.
        conversion = ConversionRequirement(
            "others_upr_hand_lay_up_conversion",
            43163838324174400 / 35680245530124351,
            67075941681801 / 205053863772800,
            (
                PrincipalExit("styrene_air", 8208310784 / 84118311615),
                PrincipalExit("post_release_solid", 103474753 / 922564095),
            ),
        )
        recipe = physical_material_recipe()
        recipe = replace(
            recipe,
            conversion=tuple(
                conversion if row.operation == conversion.operation else row
                for row in recipe.conversion
            ),
        )
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=ordinary,
            technology_database="dated",
            provider=provider,
            material_recipe=recipe,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
        for declaration in conversions:
            root = bd.get_node(
                database=frontier.database_name,
                code=provider_service_code(declaration["provider_service_id"]),
            )
            reference = root["upr_reference_per_retained_principal_kg"]
            origin = declaration["construction"]["principal_carbon_origin"]
            assert tuple(reference["retained"]["principal_by_origin_kg"]) == tuple(
                float(item == origin) for item in ("fossil", "biogenic", "atmospheric_dac")
            )
            tech = {row.input.key: dict(row) for row in root.technosphere()}
            bio = {row.input.key[1]: dict(row) for row in root.biosphere()}
            operation_amount = 205053863772800 / 67075941681801
            assert reference["operation_kg"] == pytest.approx(operation_amount)
            assert len(tech) == 28
            assert len(bio) == 5
            for label, amount in reference["supply_amounts"].items():
                selector = declaration["construction"]["supplies"][label]
                key = (
                    (frontier.database_name, provider_service_code(selector["service_id"]))
                    if "service_id" in selector
                    else (ordinary, "supply_" + label)
                )
                assert tech[key]["amount"] == pytest.approx(amount)
            residue_key = (
                frontier.database_name,
                provider_service_code(declaration["construction"]["actual_residue"]["service_id"]),
            )
            assert tech[residue_key]["amount"] == pytest.approx(103474753 / 922564095)
            original_rows = before[ordinary, "operation"][1]
            native_output = next(
                row["amount"] for row in original_rows if row["type"] == "production"
            )
            for original in original_rows:
                if original["type"] == "production":
                    continue
                actual = (
                    bio[original["input"][1]]
                    if original["type"] == "biosphere"
                    else tech[original["input"]]
                )
                scale = operation_amount / native_output
                if original["input"][1] == "residue_market":
                    scale *= 0.08803 / 0.186
                elif original["input"][1] == "styrene_air":
                    scale = (0.0975805460952249 + 0.0002447467923130798) / original["amount"]
                assert actual["amount"] == pytest.approx(original["amount"] * scale)
                assert actual["loc"] == pytest.approx(original["loc"] + log(scale))
                assert actual["scale"] == original["scale"]
                assert actual["minimum"] == pytest.approx(original["minimum"] * scale)
                assert actual["maximum"] == pytest.approx(original["maximum"] * scale)
                assert actual["negative"] == original["negative"]
                assert actual["allocation"] == original["allocation"]
            assert bio["nmvoc_air"]["amount"] == pytest.approx(0.0007282109900298825)
            assert bio["water_air"]["amount"] == pytest.approx(0.0005582950923562433 / 1000)
            lca = LCA({root: 1})
            lca.lci()
            for label, _, amount, _ in operation_inputs:
                if label in {"scrap", "mixed_waste"}:
                    continue
                index = lca.dicts.biosphere[
                    bd.get_node(database="biosphere", code="operation_" + label).id
                ]
                expected = abs(amount) * factor * operation_amount * 2 * factor
                assert float(lca.inventory[index, :].sum()) == pytest.approx(expected)
            for label, amount in reference["supply_amounts"].items():
                index = lca.dicts.biosphere[
                    bd.get_node(database="biosphere", code="supply_" + label).id
                ]
                assert float(lca.inventory[index, :].sum()) == pytest.approx(
                    amount * 1.02 * 2 * factor
                )
            index = lca.dicts.biosphere[bd.get_node(database="biosphere", code="freight").id]
            assert float(lca.inventory[index, :].sum()) == pytest.approx(
                fsum(reference["supply_amounts"].values()) * 0.028 * 2 * factor
            )
            residue = bd.get_node(database=residue_key[0], code=residue_key[1])
            residue_rows = {row.input.key: dict(row) for row in residue.technosphere()}
            assert len(residue_rows) == 4
            assert residue["whole_material_kg_per_principal_kg"] == pytest.approx(2.659625150379939)
            residue_declaration = services[
                declaration["construction"]["actual_residue"]["service_id"]
            ]
            market_output = 3 if year == 2025 else 7
            for original in before[ordinary, "residue_market"][1]:
                if original["type"] == "production":
                    continue
                code = original["input"][1]
                if code == "residue_freight":
                    actual = residue_rows[ordinary, code]
                    scale = 2.659625150379939 / market_output
                    assert actual["amount"] == pytest.approx(original["amount"] * scale)
                    assert actual["loc"] == pytest.approx(original["loc"] + log(scale))
                    continue
                role = code.removeprefix("donor_")
                target = residue_declaration["construction"]["treatments"][role]["service_id"]
                actual = residue_rows[frontier.database_name, provider_service_code(target)]
                assert actual["amount"] == pytest.approx(treatment_shares[role])
                assert actual["loc"] == pytest.approx(
                    -original["loc"] / market_output
                    if year == 2025
                    else original["loc"] - log(market_output)
                )
                if year == 2025 and role == "open_burning":
                    assert {"negative", "minimum", "maximum"}.isdisjoint(actual)
                else:
                    assert actual["negative"] is False
                    assert actual["minimum"] == pytest.approx(-original["maximum"] / market_output)
                    assert actual["maximum"] == pytest.approx(-original["minimum"] / market_output)
                assert actual["scale"] == original["scale"]
                assert actual["allocation"] == original["allocation"]
                assert (
                    not {"name", "activity", "flow", "properties", "classifications"}
                    & actual.keys()
                )
                index = lca.dicts.biosphere[
                    bd.get_node(database="biosphere", code="terminal_" + role).id
                ]
                assert float(lca.inventory[index, :].sum()) == pytest.approx(
                    (103474753 / 922564095) * treatment_shares[role] * 2 * factor
                )
                index = lca.dicts.biosphere[
                    bd.get_node(database="biosphere", code="donor_" + role).id
                ]
                assert float(lca.inventory[index, :].sum()) == pytest.approx(
                    0.08803 * operation_amount * factor * treatment_shares[role] * 2 * factor
                )
            index = lca.dicts.biosphere[
                bd.get_node(database="biosphere", code="residue_freight").id
            ]
            assert float(lca.inventory[index, :].sum()) == pytest.approx(
                (0.08803 * operation_amount * factor + (103474753 / 922564095) * 2.659625150379939)
                * 0.075154404308835
                * factor
                * 2
                * factor
            )
        after = {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
        assert after == before
