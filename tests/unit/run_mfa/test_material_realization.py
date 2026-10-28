"""Treatment and polymer balances through the common material owner."""

import json
from fractions import Fraction
from importlib.resources import files
from math import fsum

import numpy as np
import pandas as pd
import pytest
from material_recipe_fixtures import material_source_members

from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.run_mfa.model.materials import (
    MaterialRealization,
    balanced_primary_trade,
    load_material_composition,
    material_realization_table,
    realize_materials,
)
from pytoce.run_mfa.model.recycling_allocation import _dot
from pytoce.run_mfa.scenario.catalogue import _catalogue_payload
from pytoce.run_mfa.scenario.scenario_contract import ScenarioFeasibilityError

# LBNL's 2019 finished-output pools are 9.5 Mt foam, 9.9 Mt solid, and 1.2 Mt other PS. Synthetic
# yields are 1, .8, and .6 kg finished/kg feed. Exact rational normalization preserves source values
# through the receiving checks.
_PS_GROSS_FACTOR = (
    Fraction(9.5) + Fraction(9.9) * Fraction(1 / 0.8) + Fraction(1.2) * Fraction(1 / 0.6)
) / (Fraction(9.5) + Fraction(9.9) + Fraction(1.2))


def _realize(
    materials: tuple[str, ...],
    waste: tuple[float, ...],
    routes: tuple[Fraction | float, ...],
    production: tuple[Fraction | float, ...],
    imports: tuple[Fraction | float, ...] | None = None,
    exports: tuple[Fraction | float, ...] | None = None,
) -> MaterialRealization:
    """Call the real mass owner with admitted public composition and invented XML."""
    size = len(materials)
    return realize_materials(
        year=2050,
        region="EUR",
        materials=materials,
        composition=load_material_composition(),
        recipe=derive_material_recipe(material_source_members(), source_identity="invented"),
        collected=np.array(waste),
        route_totals=tuple(Fraction(value) for value in routes),
        processing_requirement=tuple(Fraction(value) for value in production),
        primary_imports=tuple(Fraction(value) for value in imports or (0.0,) * size),
        primary_exports=tuple(Fraction(value) for value in exports or (0.0,) * size),
    )


@pytest.mark.parametrize(
    ("material", "weights", "authority", "required_sources"),
    [
        (
            "BioPlastics",
            {
                "bio_pa": 0.119,
                "bio_pe": 0.097,
                "bio_pet": 0.263,
                "other_bio_based": 0.092,
                "other_biodegradable": 0.015,
                "pbat": 0.05,
                "pbs": 0.049,
                "pha": 0.024,
                "pla": 0.103,
                "starch_blends": 0.188,
            },
            "accepted_bioplastics_composition_authority",
            {
                "oecd_global_plastics_outlook_annex",
                "european_bioplastics_market_data_2017",
                "european_bioplastics_market_data_2025",
            },
        ),
        (
            "Fibres",
            {
                "polypropylene": 0.5642138274879658,
                "polyethylene_terephthalate": 0.353411962583902,
                "polyamide_6_resin": 0.047199410872872466,
                "polyamide_66_resin": 0.03517479905525972,
            },
            "accepted_fibre_composition_authority",
            {
                "oecd_global_plastics_outlook_annex",
                "unep_ryberg_plastics_value_chain",
                "murphy_2019_fibers_update",
            },
        ),
        (
            "Others",
            {
                "adhesives": 0.12870827684465494,
                "general_coatings": 0.038775664313868856,
                "marine_coatings": 0.006581245355026663,
                "non_tyre_elastomers": 0.108715495226123,
                "other_thermoplastics": 0.14748680072933834,
                "pa": 0.036871700182334585,
                "pc": 0.036871700182334585,
                "road_marking_coatings": 0.008295774234176085,
                "sealants": 0.02522079120278648,
                "thermosets": 0.4624725517293565,
            },
            "accepted_others_composition_authority",
            {"oecd_global_plastics_outlook_annex", "unep_ryberg_plastics_value_chain"},
        ),
    ],
)
def test_physical_composition_owns_family_shares_once(
    material: str,
    weights: dict[str, float],
    authority: str,
    required_sources: set[str],
) -> None:
    """Apply source-backed constituent shares to waste and new polymer before LCI."""
    state = _realize((material,), (1.0,), (0.0, 0.0, 0.0, 1.0), (2.0,))
    total_weight = sum((Fraction(value) for value in weights.values()), Fraction())
    shares = {name: Fraction(value) / total_weight for name, value in weights.items()}
    assert {item.constituent: item.share for item in state.constituents} == shares
    assert sum(shares.values()) == 1
    for flow, total in (
        ("collected_after_trade", 1.0),
        ("landfill_treatment", 1.0),
        ("manufacturing_finished_output", 2.0),
    ):
        assert dict(
            zip((item.constituent for item in state.constituents), state.amounts[flow], strict=True)
        ) == pytest.approx({key: total * float(share) for key, share in shares.items()})
    resource = json.loads(
        files("pytoce.run_mfa.data.catalogues").joinpath("material_constituents.json").read_bytes()
    )
    source = next(row for row in resource["source_authorities"] if row["source_id"] == authority)
    assert required_sources <= set(source["direct_source_authority_ids"])


@pytest.mark.parametrize(
    ("material", "expected_gross"),
    [
        ("ABS_ASA_SAN", 1 / 0.8),
        ("HDPE", 0.67 / 0.8 + 0.09 / 0.75 + 0.24 / 0.9),
        ("LDPE", 1 / 0.6),
        ("PP", 0.74 / 0.8 + 0.02 / 0.75 + 0.24 / 0.9),
        ("PET", 0.5 / 0.9 + 0.5 / 0.8),
        ("PS", (9.5 + 9.9 / 0.8 + 1.2 / 0.6) / 20.6),
        ("PVC", 0.29 / 0.8 + 0.51 / 0.75 + 0.02 / 0.9 + 0.18 / 0.5),
    ],
)
def test_finished_demand_requires_source_weighted_gross_resin(
    material: str, expected_gross: float
) -> None:
    """Use finished-output operation weights and the invented native dry yields."""
    state = _realize((material,), (0.0,), (0.0,) * 4, (1.0,))
    assert state.amounts["manufacturing_principal_input"] == pytest.approx([expected_gross])
    assert state.amounts["manufacturing_finished_output"] == pytest.approx([1.0])
    assert state.amounts["manufacturing_principal_loss"] == pytest.approx([expected_gross - 1])
    assert state.amounts["new_polymer_domestic"] == pytest.approx([expected_gross])


def test_bioplastic_processing_stays_conditional_on_the_actual_constituent() -> None:
    """Apply PE/PET operations only to their constituents and injection to the rest."""
    state = _realize(("BioPlastics",), (0.0,), (0.0,) * 4, (2.0,))
    # These are conditional finished-output assumptions, independent of the
    # family weights checked above. The invented source has different yields
    # for injection, pipe and blow, exposing a repeated family allocation.
    polyethylene = {
        "conversion_injection": 0.67,
        "conversion_pipe": 0.09,
        "conversion_blow": 0.24,
    }
    pet = {"conversion_blow": 0.5, "conversion_injection": 0.5}
    source_finished = {
        "conversion_injection": 0.8,
        "conversion_pipe": 0.75,
        "conversion_blow": 0.9,
    }
    for index, item in enumerate(state.constituents):
        expected = {"bio_pe": polyethylene, "bio_pet": pet}.get(
            item.constituent, {"conversion_injection": 1.0}
        )
        actual = {
            component.operation: float(component.share) for component in item.conversion.components
        }
        assert actual == pytest.approx(expected)
        finished = 2 * float(item.share)
        gross = finished * fsum(
            share / source_finished[operation] for operation, share in expected.items()
        )
        assert state.amounts["manufacturing_finished_output"][index] == pytest.approx(finished)
        assert state.amounts["manufacturing_principal_input"][index] == pytest.approx(gross)
        assert state.amounts["manufacturing_principal_loss"][index] == pytest.approx(
            gross - finished
        )
    assert fsum(state.amounts["manufacturing_finished_output"]) == 2.0


def test_mixed_fibres_keep_each_manufacturing_feed_boundary() -> None:
    """Keep PP retention, PET feed and both PA cleaning deposits separate in one family."""
    state = _realize(("Fibres",), (0.0,), (0.0,) * 4, (1.0,))
    gross = dict(
        zip(
            (item.constituent for item in state.constituents),
            state.amounts["manufacturing_principal_input"],
            strict=True,
        )
    )
    factors = {
        "polypropylene": 1.0,
        "polyethylene_terephthalate": 1.25,
        "polyamide_6_resin": 1.0001035 * 100000 / 99999,
        "polyamide_66_resin": 100000 / 99999,
    }
    for item in state.constituents:
        assert gross[item.constituent] == pytest.approx(
            float(item.share) * factors[item.constituent]
        )
    assert fsum(state.amounts["manufacturing_finished_output"]) == 1.0


def test_all_materials_have_a_physical_zero_state() -> None:
    """Retain zero material and monomer flows for zero waste, production, and trade."""
    materials = tuple(_catalogue_payload()["coordinate_sets"]["material"])
    state = _realize(materials, (0.0,) * len(materials), (0.0,) * 4, (0.0,) * len(materials))
    assert not material_realization_table(state)["amount"].any()
    np.testing.assert_array_equal(state.paired_trade_reduction, 0.0)


@pytest.mark.parametrize("material", ("PUR", "Elastomers (tyres)"))
def test_cured_material_is_not_unrestricted_mechanical_resin(material: str) -> None:
    """Cured products can enter terminals but cannot supply the resin-return function."""
    terminal = _realize((material,), (1.0,), (0.0, 0.0, 0.5, 0.5), (1.0,))
    assert terminal.amounts["incineration_treatment"] == pytest.approx([0.5])
    assert terminal.amounts["landfill_treatment"] == pytest.approx([0.5])
    with pytest.raises(ScenarioFeasibilityError) as error:
        _realize((material,), (1.0,), (0.25, 0.0, 0.25, 0.5), (1.0,))
    assert error.value.code == "optimization_material_recovery_capacity_infeasible"
    assert error.value.diagnostic.shortfall == 0.25


def test_receiving_capacity_preserves_exact_bounds_and_shortfalls() -> None:
    """Recovered PS meets gross receiving demand after its three forming losses."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented")
    eta = next(
        row.recovered_kg_per_input_kg
        for row in recipe.mechanical_recovery
        if row.name == "ps_granulate"
    )
    finished = Fraction(eta) / _PS_GROSS_FACTOR
    state = _realize(("PS",), (1.0,), (1.0, 0.0, 0.0, 0.0), (finished,))
    np.testing.assert_array_equal(state.amounts["new_polymer_domestic"], 0.0)
    np.testing.assert_array_equal(state.amounts["mechanical_resin"], [eta])
    # First cross the boundary by one float step; then use an exact material
    # residual smaller than a float ULP. The latter's two diagnostic operands
    # round to the same value, but its positive shortfall must survive.
    for requested in (
        Fraction(float(np.nextafter(float(finished), 0.0))),
        (Fraction(eta) - Fraction(1, 2**60)) / _PS_GROSS_FACTOR,
    ):
        with pytest.raises(ScenarioFeasibilityError) as error:
            _realize(("PS",), (1.0,), (1.0, 0.0, 0.0, 0.0), (requested,))
        diagnostic = error.value.diagnostic
        assert diagnostic is not None
        assert error.value.code == "optimization_material_recovery_capacity_infeasible"
        # The active constraint is -mechanical_input >= -receiving/eta. Requested mechanical input
        # is 1 Mt.
        required = -requested * _PS_GROSS_FACTOR / Fraction(eta)
        assert diagnostic.observed == float(required)
        assert diagnostic.bound == -1.0
        assert diagnostic.shortfall == float(required + 1) > 0
        assert diagnostic.tolerance == 0.0
        assert diagnostic.equation.endswith("required <= maximum")


def test_compatible_waste_capacity_excludes_the_cured_family() -> None:
    """An additional 1 Mt of PUR cannot cover even a one-step excess of recyclable PS."""
    state = _realize(("PS", "PUR"), (1.0, 1.0), (1.0, 0.0, 0.0, 1.0), (2.0, 2.0))
    assert fsum(state.amounts["mechanical_treatment"]) == 1.0
    requested = Fraction(float(np.nextafter(1.0, 2.0)))
    with pytest.raises(ScenarioFeasibilityError) as error:
        _realize(("PS", "PUR"), (1.0, 1.0), (requested, 0.0, 0.0, 2 - requested), (2.0, 2.0))
    assert error.value.code == "optimization_material_recovery_capacity_infeasible"
    assert error.value.diagnostic.shortfall == float(requested - 1)


def test_chemical_eligibility_accepts_exact_available_constituent() -> None:
    """The chemical share cannot borrow any of the nine other bioplastic constituents."""
    composition = load_material_composition()
    available = next(
        item.share
        for item in composition.constituents
        if item.material == "BioPlastics" and item.constituent == "bio_pe"
    )
    state = _realize(("BioPlastics",), (1.0,), (0.0, available, 0.0, 1 - available), (3.0,))
    chemical = {
        item.constituent: amount
        for item, amount in zip(
            state.constituents, state.amounts["chemical_treatment"], strict=True
        )
    }
    assert chemical["bio_pe"] == float(available)
    assert all(amount == 0 for key, amount in chemical.items() if key != "bio_pe")
    increment = Fraction(1, 2**54)
    with pytest.raises(ScenarioFeasibilityError) as error:
        _realize(
            ("BioPlastics",),
            (1.0,),
            (0.0, available + increment, 0.0, 1 - available - increment),
            (3.0,),
        )
    assert error.value.code == "optimization_chemical_feed_eligibility_infeasible"
    assert error.value.diagnostic.shortfall == float(increment)


def test_selective_treatment_preserves_all_principal_and_product_balances() -> None:
    """Apply different recovery outlets while retaining the four requested route totals."""
    totals = (Fraction(3, 5), Fraction(1, 10), Fraction(4, 5), Fraction(3, 2))
    state = _realize(("BioPlastics", "Others", "PP"), (1.0, 1.0, 1.0), totals, (3.0, 3.0, 3.0))
    amounts = state.amounts
    treatment_names = (
        "mechanical_treatment",
        "chemical_treatment",
        "incineration_treatment",
        "landfill_treatment",
    )
    for name, total in zip(treatment_names, totals, strict=True):
        assert fsum(amounts[name]) == pytest.approx(float(total), rel=1e-14)
    np.testing.assert_allclose(
        sum(amounts[name] for name in treatment_names),
        amounts["collected_after_trade"],
        rtol=1e-14,
        atol=1e-16,
    )
    mechanical_outlets = (
        "mechanical_resin",
        "mechanical_waterborne_loss",
        "mechanical_airborne_loss",
        "mechanical_incinerated_reject",
        "mechanical_landfilled_reject",
    )
    np.testing.assert_allclose(
        sum(amounts[name] for name in mechanical_outlets),
        amounts["mechanical_treatment"],
        rtol=1e-14,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        amounts["manufacturing_finished_output"] + amounts["manufacturing_principal_loss"],
        amounts["manufacturing_principal_input"],
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        amounts["chemical_processed_feed"] + amounts["chemical_preparation_reject"],
        amounts["chemical_treatment"],
        rtol=1e-14,
    )
    assert amounts["mechanical_waterborne_loss"].sum() > 0
    assert amounts["mechanical_airborne_loss"].sum() > 0
    assert amounts["mechanical_landfilled_reject"].sum() > 0
    for index, item in enumerate(state.constituents):
        if item.mechanical_recovery is None:
            assert amounts["mechanical_treatment"][index] == 0
        if item.chemical_polymer is None:
            assert amounts["chemical_treatment"][index] == 0
        if item.mechanical_recovery == "tps_compound":
            assert amounts["mechanical_incinerated_reject"][index] == 0
        else:
            assert amounts["mechanical_landfilled_reject"][index] == 0
    recipe = derive_material_recipe(material_source_members(), source_identity="invented")
    for product in recipe.hydrothermal_products:
        np.testing.assert_allclose(
            state.hydrothermal_products[product.product],
            amounts["chemical_processed_feed"] * product.kg_per_kg_feed,
        )
    for monomer in ("ethylene", "propylene"):
        assert fsum(state.recovered_monomers[monomer]) == pytest.approx(
            fsum(state.cracker_products[monomer]), rel=1e-14
        )
    table = material_realization_table(state)
    assert not table.duplicated(list(table.columns[:8])).any()
    assert set(table["unit"]) == {"Mt"}
    assert np.isfinite(table["amount"]).all() and table["amount"].ge(0).all()
    supply = table.loc[table["flow_name"].eq("cracker_product")]
    assert supply["source_constituent"].notna().all()
    assert supply["destination_constituent"].isna().all()
    consumption = table.loc[table["flow_name"].eq("recovered_monomer_consumption")]
    assert consumption["source_constituent"].isna().all()
    assert consumption["destination_constituent"].notna().all()


def test_receiving_and_monomer_limits_jointly_redirect_mechanical_recovery() -> None:
    """PET receiving capacity and propylene absorption bind at the analytic allocation."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented")
    recovery = {
        row.name: Fraction(row.recovered_kg_per_input_kg) for row in recipe.mechanical_recovery
    }
    eta_h, eta_p, eta_t = (
        recovery[name] for name in ("hdpe_granulate", "pp_granulate", "pet_granulate")
    )
    # The two-active-limit solution uses injection .8, pipe .75, and blow .9 finished-operation
    # fractions, with normalized binary source weights.
    h_weights = tuple(Fraction(x) for x in (0.67, 0.09, 0.24))
    p_weights = tuple(Fraction(x) for x in (0.74, 0.02, 0.24))
    input_factors = tuple(Fraction(1 / x) for x in (0.8, 0.75, 0.9))
    gh = (
        Fraction(0.5)
        * sum(w * g for w, g in zip(h_weights, input_factors, strict=True))
        / sum(h_weights)
    )
    gp = (
        Fraction(0.04)
        * sum(w * g for w, g in zip(p_weights, input_factors, strict=True))
        / sum(p_weights)
    )
    gt = Fraction(0.23) * (Fraction(1 / 0.9) + Fraction(1 / 0.8)) / 2
    processed = Fraction(recipe.preparation_processed_kg_per_bale_kg)
    processed /= processed + Fraction(recipe.preparation_reject_kg_per_bale_kg)
    liquids = sum(
        Fraction(row.kg_per_kg_feed)
        for row in recipe.hydrothermal_products
        if row.product in ("naphtha", "distillate_gas_oil")
    )
    propylene = next(
        Fraction(row.kg_per_kg_feed)
        for row in recipe.cracker_products
        if row.product == "propylene"
    )
    supply = Fraction(1, 2) * processed * liquids * propylene
    bh, bp = Fraction(0.02) * eta_h, Fraction(0.95) * eta_p
    pet_input = gt / eta_t
    pp_input = (Fraction(0.02) * gh + Fraction(0.95) * gp - supply - bh * (1 - pet_input)) / (
        bp - bh
    )
    hdpe_input = 1 - pet_input - pp_input
    assert 0 < pp_input < gp / eta_p
    assert 0 < hdpe_input < gh / eta_h
    state = _realize(
        ("HDPE", "PP", "PET"), (2.0, 2.0, 2.0), (1.0, 0.5, 0.0, 4.5), (0.5, 0.04, 0.23)
    )
    actual = {
        item.material: value
        for item, value in zip(
            state.constituents, state.amounts["mechanical_treatment"], strict=True
        )
    }
    assert actual == pytest.approx(
        {"HDPE": float(hdpe_input), "PP": float(pp_input), "PET": float(pet_input)}, rel=1e-13
    )
    np.testing.assert_array_equal(state.virgin_monomers["propylene"], 0.0)
    assert state.virgin_monomers["ethylene"].sum() > 0
    assert fsum(state.recovered_monomers["propylene"]) == pytest.approx(float(supply), rel=1e-14)
    pet = next(i for i, item in enumerate(state.constituents) if item.material == "PET")
    assert state.amounts["new_polymer_domestic"][pet] == 0.0


def test_unabsorbed_chemical_products_cannot_be_hidden_in_export() -> None:
    """A positive chemical request with no resin demand remains infeasible in the MFA."""
    with pytest.raises(ScenarioFeasibilityError) as error:
        _realize(("HDPE",), (1.0,), (0.0, 1.0, 0.0, 0.0), (0.0,))
    assert error.value.code == "optimization_recovered_monomer_surplus_infeasible"
    assert error.value.diagnostic.shortfall > 0


@pytest.mark.parametrize(
    ("material", "monomer", "weights", "use"),
    [
        ("HDPE", "propylene", (0.67, 0.09, 0.24), 0.02),
        ("PP", "ethylene", (0.74, 0.02, 0.24), 0.1),
    ],
)
def test_each_monomer_absorption_endpoint_preserves_a_small_virgin_complement(
    material: str, monomer: str, weights: tuple[float, ...], use: float
) -> None:
    """Full chemical treatment admits exact demand and preserves positive make-up."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented")
    processed = Fraction(recipe.preparation_processed_kg_per_bale_kg)
    processed /= processed + Fraction(recipe.preparation_reject_kg_per_bale_kg)
    liquids = sum(
        Fraction(row.kg_per_kg_feed)
        for row in recipe.hydrothermal_products
        if row.product in ("naphtha", "distillate_gas_oil")
    )
    cracker = next(
        Fraction(row.kg_per_kg_feed) for row in recipe.cracker_products if row.product == monomer
    )
    source_weights = tuple(Fraction(value) for value in weights)
    gross_factor = sum(
        weight * Fraction(1 / finished)
        for weight, finished in zip(source_weights, (0.8, 0.75, 0.9), strict=True)
    ) / sum(source_weights)
    demand = processed * liquids * cracker / (Fraction(use) * gross_factor)
    exact = _realize((material,), (1.0,), (0.0, 1.0, 0.0, 0.0), (demand,))
    np.testing.assert_array_equal(exact.virgin_monomers[monomer], 0.0)
    assert exact.amounts["chemical_treatment"] == pytest.approx([1.0])
    for terminal in ("incineration_treatment", "landfill_treatment"):
        np.testing.assert_array_equal(exact.amounts[terminal], 0.0)
    increased = _realize((material,), (1.0,), (0.0, 1.0, 0.0, 0.0), (demand + Fraction(1, 2**60),))
    assert 0 < increased.virgin_monomers[monomer][0] < 1e-16
    assert increased.recovered_monomers[monomer] == pytest.approx(exact.recovered_monomers[monomer])


def test_primary_trade_reduction_and_tiny_resin_complements_remain_exact() -> None:
    """Paired trade preserves net trade; a small positive resin complement is retained."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented")
    eta = Fraction(
        next(
            row.recovered_kg_per_input_kg
            for row in recipe.mechanical_recovery
            if row.name == "ps_granulate"
        )
    )
    mechanical = 1 / eta
    routes = (mechanical, Fraction(), Fraction(), 2 - mechanical)
    exact = _realize(("PS",), (2.0,), routes, (1 / _PS_GROSS_FACTOR,), (0.2,), (0.3,))
    np.testing.assert_array_equal(exact.primary_imports, [0.0])
    np.testing.assert_array_equal(exact.paired_trade_reduction, [0.2])
    assert exact.primary_exports == pytest.approx([0.3 - 0.2])
    assert exact.amounts["new_polymer_domestic"] == pytest.approx([0.0], abs=0.0)
    assert exact.amounts["new_polymer_export"] == pytest.approx([0.3 - 0.2])
    small = _realize(("PS",), (2.0,), routes, (Fraction(1.0 + 5e-13) / _PS_GROSS_FACTOR,))
    assert 0 < small.amounts["new_polymer_domestic"][0] < 1e-12
    assert small.amounts["new_polymer_domestic"][0] == 1.0 + 5e-13 - 1.0
    with pytest.raises(ScenarioFeasibilityError) as error:
        _realize(
            ("PS",),
            (2.0,),
            routes,
            (Fraction(float(np.nextafter(1.0, 0.0))) / _PS_GROSS_FACTOR,),
            (1.0,),
            (1.0,),
        )
    assert error.value.code == "optimization_material_recovery_capacity_infeasible"
    assert error.value.diagnostic.shortfall > 0
    assert error.value.diagnostic.tolerance == 0.0


def test_imported_primary_resin_cannot_exceed_gross_receiving_demand() -> None:
    """Admit exact imported supply; reject a real shortage of paired export capacity."""
    # The two source contributions retain their exact sum until realization.
    imported = Fraction(0.3) + Fraction(0.6)
    exact = _realize(("PS",), (0.0,), (0.0,) * 4, (imported / _PS_GROSS_FACTOR,), (imported,))
    np.testing.assert_array_equal(exact.amounts["new_polymer_domestic"], 0.0)
    np.testing.assert_array_equal(exact.primary_imports, [float(imported)])
    with pytest.raises(ScenarioFeasibilityError) as error:
        _realize(
            ("PS",),
            (0.0,),
            (0.0,) * 4,
            (Fraction(float(np.nextafter(0.3, 0.0))) / _PS_GROSS_FACTOR,),
            (0.3,),
        )
    assert error.value.code == "optimization_insufficient_paired_primary_trade_infeasible"
    assert error.value.diagnostic.shortfall > 0


def test_zero_chemical_recovery_keeps_both_ldpe_monomer_requirements() -> None:
    """Allocation is undone once; gross resin and recovered supply precede both monomers."""
    state = _realize(
        ("LDPE",), (1.0,), (Fraction(1, 5), 0.0, 0.0, Fraction(4, 5)), (1.0,), (0.1,), (0.2,)
    )
    gross = 1 / 0.6  # The selected film operation's independently invented dry yield.
    resin = gross - 0.1 + 0.2 - state.amounts["mechanical_resin"][0]
    assert state.virgin_monomers["ethylene"] == pytest.approx([1.08 * resin])
    assert state.virgin_monomers["propylene"] == pytest.approx([0.02 * resin])
    assert not state.recovered_monomers["ethylene"].any()
    assert not state.recovered_monomers["propylene"].any()
    repeated = _realize(
        ("LDPE",), (1.0,), (Fraction(1, 5), 0.0, 0.0, Fraction(4, 5)), (1.0,), (0.1,), (0.2,)
    )
    pd.testing.assert_frame_equal(
        material_realization_table(state), material_realization_table(repeated), check_exact=True
    )


def test_closed_primary_market_preserves_source_proportions_through_exhaustion() -> None:
    """Source rounding cannot become import-only residual after exact paired subtraction."""
    imports = np.array([[[0.7], [1.4]]])
    exports = np.array([[[0.6999999999999998], [1.3999999999999997]]])
    incoming, outgoing = balanced_primary_trade(imports, exports)
    assert incoming == outgoing
    assert incoming[1][0] == 2 * incoming[0][0]
    assert sum(row[0] for row in incoming) == sum(Fraction(float(x)) for x in exports.flat)
    recipe = derive_material_recipe(material_source_members(), source_identity="invented")
    eta = Fraction(
        next(
            row.recovered_kg_per_input_kg
            for row in recipe.mechanical_recovery
            if row.name == "ps_granulate"
        )
    )
    mechanical = 1 / eta
    for received, sent in zip(incoming, outgoing, strict=True):
        state = _realize(
            ("PS",),
            (2.0,),
            (mechanical, 0.0, 0.0, 2 - mechanical),
            (1 / _PS_GROSS_FACTOR,),
            received,
            sent,
        )
        np.testing.assert_array_equal(state.primary_imports, [0.0])
        np.testing.assert_array_equal(state.primary_exports, [0.0])
        np.testing.assert_array_equal(state.amounts["new_polymer_domestic"], [0.0])
        np.testing.assert_array_equal(state.amounts["new_polymer_export"], [0.0])


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            (
                Fraction(),
                Fraction(1),
                Fraction(-1),
                Fraction(2, 3),
                Fraction(-3, 7),
                Fraction(5, 4),
            ),
            (
                Fraction(8, 9),
                Fraction(5, 6),
                Fraction(5, 6),
                Fraction(7, 4),
                Fraction(-5, 3),
                Fraction(),
            ),
            Fraction(79, 42),
        ),
        (
            (Fraction(1), Fraction(-1)),
            (Fraction(float(np.finfo(np.float64).max)),) * 2,
            Fraction(),
        ),
        (
            (Fraction(1), Fraction(-1), Fraction(2, 3)),
            (Fraction(1, 1 << 1074),) * 3,
            Fraction(1, 3 * (1 << 1073)),
        ),
        ((Fraction(), Fraction(1)), (Fraction(5, 7), Fraction()), Fraction()),
    ],
    ids=("signed_sparse_normals", "largest_finite_cancellation", "subnormal_residual", "zero"),
)
def test_material_constraint_products_preserve_exact_canonical_rationals(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...], expected: Fraction
) -> None:
    """Keep sparse receiving normals and dense monomer coefficients on one exact basis."""
    result = _dot(left, right)
    assert isinstance(result, Fraction)
    assert (result.numerator, result.denominator) == (expected.numerator, expected.denominator)


@pytest.mark.parametrize(
    ("left", "right"),
    [((Fraction(),), ()), ((), (Fraction(),))],
)
def test_zero_material_constraint_terms_do_not_hide_dimension_mismatch(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> None:
    """A skipped zero must not suppress the existing strict vector-length error."""
    with pytest.raises(ValueError) as expected:
        sum((a * b for a, b in zip(left, right, strict=True)), Fraction())
    with pytest.raises(ValueError) as observed:
        _dot(left, right)
    assert str(observed.value) == str(expected.value)
