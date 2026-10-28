"""PA embedded finish follows principal routes with its own carbon identity."""

from math import fsum

import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from pa_custody_fixtures import two_cycle_pa_mfa

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.run_lca.calculation.custody import (
    CarbonCustodyTrace,
    prepare_carbon_cohort_weights,
    trace_carbon_custody,
)
from pytoce.run_lca.calculation.providers import MaterialCarbon
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.model.mfa import MFAEvaluation

# Chain Yarn US20200002854A1 Table 2: mean of the three paired postwash /
# prewash oil-on-fiber measurements, independently for each polyamide grade.
_PA_REMAINING = (
    ("polyamide_6_resin", (0.22 / 1.21 + 0.20 / 1.23 + 0.19 / 1.21) / 3),
    ("polyamide_66_resin", (0.20 / 1.35 + 0.20 / 1.34 + 0.22 / 1.34) / 3),
)
# Haian's admitted raw-input reference: of 100,000 kg, 10 kg adheres and
# 9 kg returns internally. Only 1 kg leaves through oxidative cleaning.
_DEPOSIT_FRACTION = 1 / 100000


@pytest.fixture(scope="module")
def two_recycling_cycles(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> tuple[MFAEvaluation, CarbonCustodyTrace]:
    """Replace one annual fiber cohort twice, then incinerate its final retirement."""
    carbon, recipe = carbon_authorities
    mfa = two_cycle_pa_mfa(recipe, biogenic_share=0.0, dac_share=0.0)
    return mfa, trace_carbon_custody(mfa, carbon, recipe)


def test_two_pa_cycles_keep_inherited_finish_and_partition_actual_outlets(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
    two_recycling_cycles: tuple[MFAEvaluation, CarbonCustodyTrace],
) -> None:
    """Wash each new surface once and preserve inherited dose through all principal exits."""
    _, recipe = carbon_authorities
    _, trace = two_recycling_cycles
    amounts = trace.pa_finish_amounts
    composition = load_material_composition().constituents
    shares = {
        item.constituent: float(item.share) for item in composition if item.material == "Fibres"
    }
    recovery = next(item for item in recipe.mechanical_recovery if item.name == "pa_granulate")
    # Principal outlet fractions are admitted MFA inputs. This test checks the corresponding
    # auxiliary allocation.
    outlets = {
        "reclmech => processing": recovery.recovered_kg_per_input_kg,
        "reclmech => airborne_loss": recovery.airborne_kg_per_input_kg,
        "reclmech => incineration": 1
        - recovery.recovered_kg_per_input_kg
        - recovery.airborne_kg_per_input_kg,
    }
    for constituent, remaining in _PA_REMAINING:
        principal = shares[constituent]
        previous_entry_dose = 0.0
        for year in (2019, 2020):
            # Half the previous cohort is washed. Its old embedded dose is
            # untouched; only the new surface contributes the paired remainder.
            prewash = 0.5 * previous_entry_dose
            postwash = prewash + remaining * 0.5 * principal
            recycled = recovery.recovered_kg_per_input_kg * postwash
            finished = (1 - _DEPOSIT_FRACTION) * recycled
            key = (year, "EUR", "Fibres", constituent)
            assert amounts[(*key, "all", "collected => reclmech")] == pytest.approx(
                prewash, rel=3e-13, abs=0
            )
            actual_outlets = {flow: amounts[(*key, "all", flow)] for flow in outlets}
            assert actual_outlets == pytest.approx(
                {flow: fraction * postwash for flow, fraction in outlets.items()},
                rel=3e-13,
                abs=0,
            )
            assert fsum(actual_outlets.values()) == pytest.approx(postwash, rel=3e-13)
            assert (*key, "all", "reclmech => waterborne_loss") not in amounts
            assert amounts[(*key, "all", "processing => finished_principal")] == pytest.approx(
                finished, rel=3e-13
            )
            assert amounts[(*key, "all", "processing => spinning_equipment_deposits")] == (
                pytest.approx(_DEPOSIT_FRACTION * recycled, rel=3e-13)
            )
            assert amounts[(*key, "Packaging", "fabrication => use")] == pytest.approx(
                finished, rel=3e-13
            )
            assert amounts[(*key, "all", "collected => incineration")] == pytest.approx(
                0.25 * previous_entry_dose, rel=3e-13, abs=0
            )
            assert amounts[(*key, "all", "collected => landfill")] == pytest.approx(
                0.25 * previous_entry_dose, rel=3e-13, abs=0
            )
            assert amounts[(*key, "all", "new_polymer => processing")] == 0
            if constituent == "polyamide_6_resin":
                # Selective principal caprolactam loss cannot withdraw finish.
                for exit_name in ("caprolactam_air", "caprolactam_collected_water"):
                    assert amounts[(*key, "all", f"processing => {exit_name}")] == 0
            previous_entry_dose = finished
        assert amounts[
            (2021, "EUR", "Fibres", constituent, "all", "collected => incineration")
        ] == (pytest.approx(previous_entry_dose, rel=3e-13))
    assert {key[3] for key in amounts} == {item[0] for item in _PA_REMAINING}


def test_pa_history_preserves_zero_virgin_density_and_exact_prospective_resume(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
    two_recycling_cycles: tuple[MFAEvaluation, CarbonCustodyTrace],
) -> None:
    """Reuse 2018/2019 PA cohort densities."""
    carbon, recipe = carbon_authorities
    mfa, full = two_recycling_cycles
    history = {
        (year, constituent): density
        for year, _, _, constituent, _, density in full.history.pa_finish_entries
    }
    assert {year for year, _ in history} == {2018, 2019}
    for constituent, _ in _PA_REMAINING:
        assert history[(2018, constituent)] == 0
        assert history[(2019, constituent)] > 0
    resumed = trace_carbon_custody(
        mfa,
        carbon,
        recipe,
        history=full.history,
        cohort_weights_by_year=prepare_carbon_cohort_weights(
            mfa.cohort_state, mfa.cohort_material_identity
        ),
    )
    assert resumed.history == full.history
    assert resumed.pa_finish_amounts == {
        key: amount for key, amount in full.pa_finish_amounts.items() if key[0] > 2019
    }
    pd.testing.assert_frame_equal(
        resumed.calculation,
        full.calculation.loc[full.calculation.year > 2019].reset_index(drop=True),
    )


def test_same_year_pa_recycling_solves_embedded_feedback_without_surface_double_count(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> None:
    """Solve retained current-cohort finish with its actual simultaneous recycling flow."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[1, region, good] = 1
    state.stock_by_cohort[1, 1, region, good] = 1
    state.outflow_by_cohort[2, 1, region, good] = 1
    state.annual_inflow[2, region, good] = 1
    state.stock_by_cohort[2:, 2, region, good] = 0.5
    state.outflow_by_cohort[2, 2, region, good] = 0.5
    inputs.mechanical_recycling_rate[2, region] = 0.5
    inputs.incineration_rate[2, region] = 0.5
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    assert trace.history.pa_finish_entries
    assert all(row[-1] == 0 for row in trace.history.pa_finish_entries)
    recovery = next(
        item.recovered_kg_per_input_kg
        for item in recipe.mechanical_recovery
        if item.name == "pa_granulate"
    )
    shares = {
        item.constituent: float(item.share)
        for item in load_material_composition().constituents
        if item.material == "Fibres"
    }
    for constituent, remaining in _PA_REMAINING:
        # Principal s enters use, while old s plus current .5s retire. Half
        # is recycled. For current embedded dose z: z=k*(.25z+.75s*rho),
        # where k includes resin recovery and the one cleaning loss only.
        k = (1 - _DEPOSIT_FRACTION) * recovery
        expected = k * 0.75 * shares[constituent] * remaining / (1 - 0.25 * k)
        key = (2020, "EUR", "Fibres", constituent)
        amounts = trace.pa_finish_amounts
        assert amounts[(*key, "Packaging", "fabrication => use")] == pytest.approx(
            expected, rel=3e-13
        )
        assert amounts[(*key, "all", "collected => reclmech")] == pytest.approx(
            0.25 * expected, rel=3e-13
        )
        assert amounts[(*key, "all", "reclmech => processing")] == pytest.approx(
            expected / (1 - _DEPOSIT_FRACTION), rel=3e-13
        )
        assert amounts[(*key, "all", "collected => incineration")] == pytest.approx(
            0.25 * expected, rel=3e-13
        )


def test_pa_finished_and_waste_trade_keep_the_producing_cohort_dose(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> None:
    """Carry EUR recycled fiber through CHA use and USA waste recovery."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
    europe, china, usa = (state.regions.index(name) for name in ("EUR", "CHA", "USA"))
    family, good = state.materials.index("Fibres"), state.goods.index("Packaging")
    state.annual_inflow[0, europe, good] = 1
    state.stock_by_cohort[0, 0, europe, good] = 1
    state.outflow_by_cohort[1, 0, europe, good] = 1
    state.annual_inflow[1, china, good] = 1
    state.stock_by_cohort[1, 1, china, good] = 1
    state.outflow_by_cohort[2, 1, china, good] = 1
    state.annual_inflow[2, usa, good] = 1
    state.stock_by_cohort[2, 2, usa, good] = 1
    state.outflow_by_cohort[3, 2, usa, good] = 1
    # The fixture's nominal C/Other Elements partition is .25/.75; both
    # entries jointly transfer one Mt through the real trade owner.
    inputs.final_exports[1, :, europe, family, good] = (0.25, 0.75)
    inputs.final_imports[1, :, china, family, good] = (0.25, 0.75)
    inputs.waste_exports[2, :, china, family, good] = (0.25, 0.75)
    inputs.waste_imports[2, :, usa, family, good] = (0.25, 0.75)
    for time, region in ((1, europe), (2, usa)):
        inputs.mechanical_recycling_rate[time, region] = 0.5
        inputs.incineration_rate[time, region] = 0.5
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    recovery = next(
        item.recovered_kg_per_input_kg
        for item in recipe.mechanical_recovery
        if item.name == "pa_granulate"
    )
    shares = {
        item.constituent: float(item.share)
        for item in load_material_composition().constituents
        if item.material == "Fibres"
    }
    for constituent, remaining in _PA_REMAINING:
        first = (1 - _DEPOSIT_FRACTION) * recovery * 0.5 * shares[constituent] * remaining
        second = (
            (1 - _DEPOSIT_FRACTION)
            * recovery
            * (0.5 * first + 0.5 * shares[constituent] * remaining)
        )
        amounts = trace.pa_finish_amounts
        assert amounts[
            (2019, "EUR", "Fibres", constituent, "all", "processing => finished_principal")
        ] == (pytest.approx(first, rel=3e-13))
        assert amounts[(2019, "CHA", "Fibres", constituent, "Packaging", "good_market => use")] == (
            pytest.approx(first, rel=3e-13)
        )
        assert (
            2019,
            "CHA",
            "Fibres",
            constituent,
            "all",
            "processing => finished_principal",
        ) not in amounts
        assert amounts[(2020, "USA", "Fibres", constituent, "all", "collected => reclmech")] == (
            pytest.approx(0.5 * first, rel=3e-13)
        )
        assert amounts[(2020, "USA", "Fibres", constituent, "Packaging", "fabrication => use")] == (
            pytest.approx(second, rel=3e-13)
        )
        assert amounts[
            (2021, "USA", "Fibres", constituent, "all", "collected => incineration")
        ] == (pytest.approx(second, rel=3e-13))


@pytest.mark.parametrize("family", ("Others", "BioPlastics"))
def test_uncoated_pa_families_do_not_acquire_fibre_finish(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe], family: str
) -> None:
    """Keep actual non-fiber PA manufacture and recycling outside fiber finish custody."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model(family)
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[1, region, good] = 1
    state.stock_by_cohort[1, 1, region, good] = 1
    state.outflow_by_cohort[2, 1, region, good] = 1
    state.annual_inflow[2, region, good] = 2
    state.stock_by_cohort[2:, 2, region, good] = 2
    # Five percent is within the independently admitted recyclable fraction
    # of both mixed families, including the restricted Others composition.
    inputs.mechanical_recycling_rate[2, region] = 0.05
    inputs.incineration_rate[2, region] = 0.95
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    assert trace.pa_finish_amounts == {}
    assert trace.history.pa_finish_entries == ()
    principal = "pa" if family == "Others" else "bio_pa"
    retained = trace.calculation.loc[
        trace.calculation.constituent.eq(principal)
        & trace.calculation.custody_status.eq("mechanically_recycled_output")
    ]
    assert retained.carbon_amount_mt.sum() > 0
