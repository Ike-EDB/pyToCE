"""Constituent carbon accounting through real native-property and recipe owners."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from material_carbon_fixtures import material_carbon_sources

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority,
    _service_source_closures,
    _source_authority_closure,
    load_provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.material_carbon import (
    CARBON_ORIGINS,
    prepare_material_carbon,
)
from pytoce.run_mfa.model.materials import load_material_composition


def test_constituent_carbon_uses_actual_exchange_properties_and_all_suppliers(
    tmp_path: Path,
) -> None:
    """Keep all coordinates, source dictionaries and distinct ultimate origins."""
    names = BrightwayNames(tmp_path / "brightway", "carbon", "biosphere", "native")
    data, flows = material_carbon_sources("native", "biosphere")
    catalogue = load_provider_catalogue()
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(flows, searchable=False)
        bd.Database("native").write(data, searchable=False)
        before = deepcopy(bd.Database("native").load())
        rows = prepare_material_carbon("native", catalogue)
        assert bd.Database("native").load() == before
    table = pd.DataFrame(rows)
    composition = load_material_composition().constituents
    assert list(
        table[["material", "constituent", "virgin_feedstock_route"]].itertuples(
            index=False, name=None
        )
    ) == [
        (item.material, item.constituent, route) for item in composition for route in CARBON_ORIGINS
    ]
    values = (
        table[["fossil_carbon", "biogenic_carbon", "atmospheric_dac_carbon"]]
        .to_numpy()
        .reshape(len(composition), 3, 3)
    )
    assert np.isfinite(values).all() and (values >= 0).all() and (values.sum(axis=2) <= 1).all()
    indexed = {item.constituent: values[i] for i, item in enumerate(composition)}
    np.testing.assert_array_equal(indexed["hdpe"], np.eye(3) * (6 / 7))
    np.testing.assert_array_equal(indexed["starch_blends"], indexed["other_biodegradable"])
    assert indexed["starch_blends"][0, 1] > 0
    assert indexed["pbat"][1, 0] > 0
    ps_source = json.loads(table.loc[table.constituent.eq("ps"), "source_inputs_json"].iloc[0])[
        "parameters"
    ]
    assert ps_source["finished_material_reference"]["components_kg_per_finished_kg"] == {
        "gpps_as_non_hips_ps": {
            "polystyrene": 0.9862,
            "polybutadiene": 0.0,
            "mineral_oil": 0.0138,
        },
        "hips": {"polystyrene": 0.902, "polybutadiene": 0.082, "mineral_oil": 0.016},
    }
    pbat_source = json.loads(table.loc[table.constituent.eq("pbat"), "source_inputs_json"].iloc[0])[
        "parameters"
    ]
    # Only charged monomers/brancher belong in this kg-valued source field;
    # service-linked water treatment is m3 and fixed fossil TBOT is separate.
    assert pbat_source["selected_monomer_inputs_kg"] == {
        "butanediol_delivered_biogenic": 0.4392278262574763,
        "purified_terephthalic_acid_delivered_biogenic": 0.36633657423006194,
        "adipic_acid_delivered_biogenic": 0.36633657423006194,
        "glycerine_delivered_biogenic": 0.00096,
    }
    assert pbat_source["fixed_source_inputs_kg"]["titanium_butoxide"] == 0.000672
    assert indexed["pla"][0, 1] == 0 and indexed["pla"][2, 1] > 0
    source = json.loads(table.loc[table.constituent.eq("pur"), "source_inputs_json"].iloc[0])[
        "native_formulations"
    ]
    assert source["long"]["carbon"] == 0.6
    assert source["short"]["carbon"] == 0.5
    adhesive_sources = json.loads(
        table.loc[table.constituent.eq("adhesives"), "source_inputs_json"].iloc[0]
    )["native_formulations"]
    assert set(adhesive_sources) == {"long", "adhesive"}
    thermoset_sources = json.loads(
        table.loc[table.constituent.eq("thermosets"), "source_inputs_json"].iloc[0]
    )["native_formulations"]
    assert set(thermoset_sources) == {"upr"}
    assert thermoset_sources["upr"]["carbon"] == 0.7
    for constituent in ("general_coatings", "marine_coatings"):
        epoxy = json.loads(
            table.loc[table.constituent.eq(constituent), "source_inputs_json"].iloc[0]
        )["parameters"]
        assert epoxy["epoxy_equivalent_weight_g_per_eq"] == 186
        assert epoxy["mean_repeat_units"] == pytest.approx(8 / 71)
        assert epoxy["principal_carbon_kg_per_kg"] == pytest.approx(0.7428441617446615)
    assert source["adhesive"]["polyol"] == pytest.approx(0.4)
    assert source["adhesive"]["mdi"] == pytest.approx(0.55)
    assert source["adhesive"]["additive_carbon"] == pytest.approx(0.02)
    # Independent conventional-route precursor and finished-product equations.
    long_bio = 0.6 * (0.1 * 36 / 92) / (0.2 * 24 / 44 + 0.7 * 36 / 58 + 0.1 * 36 / 92)
    short_bio = (
        0.5
        * (0.1 * 36 / 92 + 0.15 * 72 / 182 + 0.1 * 144 / 342)
        / (
            0.05 * 24 / 44
            + 0.6 * 36 / 58
            + 0.1 * 36 / 92
            + 0.15 * 72 / 182
            + 0.1 * 144 / 342
            + 0.00494 * 84 / 122
            + 0.0039 * 156 / 198
        )
    )
    expected_bio = (
        0.44 * (0.7 * long_bio / 1.02)
        + 0.32 * (0.4 * short_bio / 1.03)
        + 0.24 * (0.98 * 0.4 * long_bio)
    )
    assert indexed["pur"][0, 1] == pytest.approx(expected_bio)
    diagonal = np.eye(3)
    primer = 25 / 738
    sugar_c = 0.1 * 144 / 342
    sorbitol_c = 0.15 * 72 / 182
    amines_c = 0.00494 * 84 / 122 + 0.0039 * 156 / 198
    short_total = 0.05 * 24 / 44 + 0.6 * 36 / 58 + 0.1 * 36 / 92 + sorbitol_c + sugar_c + amines_c
    short_fossil = 0.5 * amines_c / short_total
    short_dac_bio = 0.5 * (sugar_c + sorbitol_c * primer) / short_total
    long_vectors = np.array(((0.6 - long_bio, long_bio, 0), (0, 0.6, 0), (0, 0, 0.6)))
    short_vectors = np.array(
        (
            (0.5 - short_bio, short_bio, 0),
            (short_fossil, 0.5 - short_fossil, 0),
            (short_fossil, short_dac_bio, 0.5 - short_fossil - short_dac_bio),
        )
    )
    flexible = (
        0.7 * long_vectors
        + (0.3 * 108 / 174 + 0.01 * (0.75 * 96 / 104 + 0.25 * 36 / 53) - 0.04 * 12 / 44) * diagonal
    ) / 1.02
    rigid = (0.4 * short_vectors + 0.6 * 0.72 * diagonal + np.array((0.04 * 60 / 72, 0, 0))) / 1.03
    adhesive = 0.98 * (0.4 * long_vectors + 0.55 * 0.72 * diagonal + np.array((0.02, 0, 0)))
    pur = 0.44 * flexible + 0.32 * rigid + 0.24 * adhesive
    others = (0.12870827684465494 + 0.02522079120278648) * adhesive + diagonal * (
        (0.038775664313868856 + 0.006581245355026663) * 0.7428441617446615
        + 0.108715495226123 * (0.235 * 96 / 104 + 0.765 * 48 / 54)
        + (0.14748680072933834 + 0.008295774234176085) * 0.6
        + 0.036871700182334585 * 72 / 113
        + 0.036871700182334585 * 192 / 254
        + 0.4624725517293565 * 0.7
    )
    glucose_origins = np.array(((1 - primer, primer, 0), (0, 1, 0), (0, primer, 1 - primer)))
    # kg C/kg principal TPS from recipe product masses and CIAAW (2024)
    # C6H10O5 and C3H8O3 fractions, retaining the 25/738 biogenic glucan primer.
    # Rows are fossil/biogenic/DAC feedstock; columns are those ultimate C origins.
    tps = np.array(
        (
            (0.4190546373358607, 0.010969573299714028, 0),
            (0, 0.43008615722273447, 0),
            (0, 0.010969573299714028, 0.4190546373358607),
        )
    )
    pbat = 0.6207700759259207 * diagonal + np.array((0.004858363658085175, 0, 0))
    pbs = 0.5573026976304111 * diagonal + np.array((0.0007735351731254874, 0, 0))
    # Accepted finished-PTT domain values: q=1-r backbone and included fossil
    # TBOT. Fossil/DAC PDO retains the 25/738 biogenic glucose-primer carbon.
    ptt = np.array(
        (
            (0.6348238439606325, 0.005917186307785407, 0),
            (0.0002647843137254902, 0.6404762459546925, 0),
            (0.0002647843137254902, 0.005917186307785407, 0.6345590596469071),
        )
    )
    pla = 0.5 * diagonal
    pla[2] = 0.5 * glucose_origins[2]
    # One commercial kg includes the ACC/Franklin nominal oil/PBR reference;
    # the LBNL finished-grade weights apply once. Fixed fossil oil remains
    # inside the principal product even on renewable synthesis routes.
    ps_selected = 0.650485436893 * 0.9862 * 96 / 104 + 0.349514563107 * (
        0.902 * 96 / 104 + 0.082 * 48 / 54
    )
    ps_oil = (0.650485436893 * 0.0138 + 0.349514563107 * 0.016) * 6 / 7
    ps_expected = ps_selected * diagonal + np.array((ps_oil, 0, 0))
    expected = np.array(
        (
            0.843664194433806 * diagonal,
            72 / 113 * diagonal,
            6 / 7 * diagonal,
            0.625 * diagonal,
            ptt,
            tps,
            pbat,
            pbs,
            0.558067 * glucose_origins,
            pla,
            tps,
            6 / 7 * diagonal,
            0.625 * diagonal,
            72 / 113 * diagonal,
            144 / 226 * diagonal,
            35 / 39 * diagonal,
            6 / 7 * diagonal,
            0.625 * diagonal,
            6 / 7 * diagonal,
            ps_expected,
            pur,
            6 / 7 * diagonal,
            24 / 62.5 * diagonal,
            adhesive,
            0.7428441617446615 * diagonal,
            0.7428441617446615 * diagonal,
            (0.235 * 96 / 104 + 0.765 * 48 / 54) * diagonal,
            0.6 * diagonal,
            72 / 113 * diagonal,
            192 / 254 * diagonal,
            0.6 * diagonal,
            adhesive,
            0.7 * diagonal,
        )
    )
    np.testing.assert_allclose(values, expected, rtol=2e-15, atol=2e-16)
    # The constituent shares reproduce the complete family carbon equation.
    family_carbon = sum(
        float(item.share) * indexed[item.constituent]
        for item in composition
        if item.material == "Others"
    )
    np.testing.assert_allclose(family_carbon, others, rtol=2e-15, atol=2e-16)
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    closures = _service_source_closures(catalogue)
    routes = {row["route_id"]: row for row in catalogue["routes"]}
    polymers = {"hdpe": "hdpe", "ldpe": "ldpe", "pp": "pp", "bio_pe": "hdpe", "polypropylene": "pp"}
    for row in rows:
        details = json.loads(row["source_inputs_json"])
        assert details["basis"] and details["source_authority_ids"]
        if row["constituent"] in {"other_biodegradable", "starch_blends"}:
            fractions = details["principal_element_mass_fractions"]
            # CIAAW elemental g/mol divided by formula g/mol gives kg C/kg.
            for name, molar_mass, carbon_mass in (
                ("glucan", 162.141, 72.066),
                ("glycerol", 92.094, 36.033),
            ):
                assert sum(fractions[name].values()) == pytest.approx(1.0, rel=2e-15)
                assert fractions[name]["C"] == carbon_mass / molar_mass
            assert "ciaaw_2024_abridged_atomic_weights" in details["source_authority_ids"]
        assert details["provider_authority_identity"].startswith("pytoce_provider_services_")
        components = details["provider_components"]
        component_sources = set()
        for component in components:
            service_id = component["provider_service_id"]
            assert component["method_authority_id"] == services[service_id]["citation"]
            component_sources.update(closures[service_id])
            route = routes[component["route_id"]]
            assert route["material"] == row["material"]
            assert route["constituent"] == row["constituent"]
            assert route["provider_service_id"] == service_id
            route_method = route.get("method_authority_id", services[service_id]["citation"])
            assert component["route_method_authority_id"] == route_method
            component_sources.update(
                _source_authority_closure(
                    catalogue,
                    json.loads(
                        _method_authority(catalogue, route_method)["source_authority_ids_json"]
                    ),
                )
            )
        assert details["source_authority_ids"] == sorted(component_sources)
        if row["constituent"] in polymers:
            polymer = polymers[row["constituent"]]
            route = row["virgin_feedstock_route"]
            assert [(c["role"], c["provider_service_id"]) for c in components] == [
                ("polymerization", f"{polymer}_polymerization_operation"),
                ("ethylene_supply", f"{polymer}_ethylene_virgin_{route}"),
                ("propylene_supply", f"{polymer}_propylene_virgin_{route}"),
            ]
        else:
            assert len(components) == 1 and components[0]["role"] == "constituent_product"
    fossil_pa = table.constituent.eq("bio_pa") & table.virgin_feedstock_route.eq("fossil")
    assert (
        json.loads(table.loc[fossil_pa, "source_inputs_json"].iloc[0])["provider_components"][0][
            "provider_service_id"
        ]
        == "nylon_6_fossil"
    )


def test_invalid_native_carbon_inputs_fail_without_changing_the_database(tmp_path: Path) -> None:
    """Reject actual missing properties, units, quantities and wrong source signs."""
    names = BrightwayNames(tmp_path / "brightway", "carbon_boundary", "biosphere", "native")
    data, flows = material_carbon_sources("native", "biosphere")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(flows, searchable=False)
        bd.Database("native").write(data, searchable=False)
        catalogue = load_provider_catalogue()
        cases = (
            ("long", "production", "properties", {}),
            (
                "long",
                "production",
                "properties",
                {
                    "carbon content": {"amount": 0.0, "unit": "dimensionless"},
                    "dry mass": {"amount": 1.0, "unit": "kg"},
                },
            ),
            (
                "short",
                "production",
                "properties",
                {
                    "carbon content": {"amount": 0.5, "unit": "kg"},
                    "dry mass": {"amount": 1.0, "unit": "kg"},
                },
            ),
            ("long", "production", "amount", -1.0),
            ("long", "production", "unit", "gram"),
            ("long", "technosphere", "unit", "gram"),
            ("rigid", "biosphere", "amount", -0.01),
            ("rigid", "biosphere", "amount", 0.1),
            ("flexible", "biosphere", "amount", 1.0),
        )
        for role, kind, field, invalid in cases:
            exchange = next(
                row
                for row in bd.get_node(database="native", code=role).exchanges()
                if row["type"] == kind
            )
            old = deepcopy(exchange[field])
            exchange[field] = invalid
            exchange.save()
            before = deepcopy(bd.Database("native").load())
            with pytest.raises(ValueError, match="Native") as failure:
                prepare_material_carbon("native", catalogue)
            if kind == "technosphere":
                assert str(failure.value) == (
                    f"Native carbon source {exchange.output.key!r} lacks a valid "
                    f"{exchange.input['reference product']!r} kilogram exchange. "
                    "Use the documented ecoinvent 3.12 source."
                )
            assert bd.Database("native").load() == before
            exchange[field] = old
            exchange.save()
        mdi = next(
            row
            for row in bd.get_node(database="native", code="rigid").technosphere()
            if row.input["reference product"] == "methylene diphenyl diisocyanate"
        )
        mdi["amount"] = 2.0
        mdi.save()
        before = deepcopy(bd.Database("native").load())
        with pytest.raises(ValueError, match="Native rigid foam yields invalid retained carbon"):
            prepare_material_carbon("native", catalogue)
        assert bd.Database("native").load() == before
        for exchange in bd.get_node(database="native", code="long").technosphere():
            exchange["amount"] = 0.0
            exchange.save()
        with pytest.raises(ValueError, match="no identified carbon-forming precursor"):
            prepare_material_carbon("native", catalogue)
