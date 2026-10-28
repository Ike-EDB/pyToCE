"""Principal manufacturing losses, retained carbon and subsequent custody."""

from math import fsum

import numpy as np
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa

from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_mfa.model.materials import load_material_composition

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
_SHARES = np.array((0.5, 0.25, 0.25))


def _upr_reference():
    """Apply the invented XML's 30% styrene, 2% release and 80% composite output."""
    principal = 0.97 * 65 / 600 + 535 / 600
    total = principal * 2.5 + 0.03 * 65 / 600 + 0.02 + 0.0025
    released = 0.02 * total * (0.3 * principal) / (0.3 * principal + 0.001)
    trim_retention = 0.8 * total / (total - 0.02 * total - 0.0006 - 0.00046)
    finished = trim_retention * (principal - released)
    # The native-shaped carbon fixture has 0.7 kg C/kg UPR, deliberately
    # different from the licensed source. Accelerator styrene is auxiliary.
    incoming_carbon = 0.7
    remaining_carbon = principal * incoming_carbon - released * 12 / 13
    losses = {
        "styrene_air": (released / finished, released * 12 / 13 / finished),
        "post_release_solid": (
            (1 - trim_retention) * (principal - released) / finished,
            (1 - trim_retention) * remaining_carbon / finished,
        ),
    }
    return (
        principal / finished,
        incoming_carbon,
        remaining_carbon * trim_retention / finished,
        losses,
    )


def _sbr_reference():
    """Use EPA compound-22 principal withdrawals before the 5% finishing transfer."""
    styrene = 0.000003657420773044308
    butadiene = 0.0000003879806170028767
    dust = 0.0000266
    before_trim = 1 - styrene - butadiene - dust
    finished = 0.95 * before_trim
    incoming_carbon = 0.235 * 12 / 13 + 0.765 * 8 / 9
    carbon_before_trim = (
        incoming_carbon - styrene * 12 / 13 - butadiene * 8 / 9 - dust * incoming_carbon
    )
    losses = {
        "styrene_air": (styrene / finished, styrene * 12 / 13 / finished),
        "butadiene_air": (butadiene / finished, butadiene * 8 / 9 / finished),
        # The admitted dust transfer uses the original charge composition.
        "extrusion_dust": (dust / finished, dust * incoming_carbon / finished),
        "post_release_solid": (0.05 * before_trim / finished, 0.05 * carbon_before_trim / finished),
    }
    return 1 / finished, incoming_carbon, 0.95 * carbon_before_trim / finished, losses


def _epoxy_reference():
    """Keep GY250 principal through preparation and airless deposition, excluding MXDA."""
    preparation = 0.08608375819937440
    finished = (1 - preparation) * 0.675
    repeats = (2 * 186 - 340) / 284
    carbon = (252 + 216 * repeats) / (340 + 284 * repeats)
    mass = {
        "preparation_residue": preparation / finished,
        "settled_overspray": 0.95 * (1 - preparation) * 0.325 / finished,
        "airborne_overspray": 0.05 * (1 - preparation) * 0.325 / finished,
    }
    return (
        1 / finished,
        carbon,
        carbon,
        {name: (value, value * carbon) for name, value in mass.items()},
    )


def _road_reference():
    """Remove Silikal's homogeneous container residue before the MMA air release."""
    principal = 0.223
    container = 0.00223
    released = 0.00202
    finished = 0.21875
    carbon = 0.6
    return (
        principal / finished,
        carbon,
        carbon,
        {
            "container_residue": (container / finished, container * carbon / finished),
            "methyl_methacrylate_air": (released / finished, released * carbon / finished),
        },
    )


@pytest.fixture(scope="module")
def manufactured_cases(carbon_authorities):
    """Manufacture one mixed-origin Mt in 2019 and retire that same cohort in 2020."""
    carbon, recipe = carbon_authorities
    cases = {}
    for family in ("Others", "Fibres", "HDPE", "BioPlastics", "PS"):
        state, inputs = empty_model(family)
        region, good = state.regions.index("EUR"), state.goods.index("Packaging")
        state.annual_inflow[1, region, good] = 1
        state.stock_by_cohort[1, 1, region, good] = 1
        state.outflow_by_cohort[2, 1, region, good] = 1
        inputs.bio_production_rate[1, region] = 0.25
        inputs.daccu_production_rate[1, region] = 0.25
        mfa = evaluate_mfa(state, inputs, recipe)
        cases[family] = (mfa, trace_carbon_custody(mfa, carbon, recipe))
    return cases


@pytest.mark.parametrize(
    ("constituent", "reference"),
    (
        pytest.param("thermosets", _upr_reference, id="upr-styrene-before-trim"),
        pytest.param("non_tyre_elastomers", _sbr_reference, id="sbr-monomers-and-charge-dust"),
        pytest.param("general_coatings", _epoxy_reference, id="epoxy-principal-without-hardener"),
        pytest.param("road_marking_coatings", _road_reference, id="road-container-before-mma"),
    ),
)
def test_principal_manufacturing_exits_and_retired_carbon(
    manufactured_cases, constituent, reference
) -> None:
    """Close total principal mass and its carbon-origin accounts before cohort entry."""
    mfa, trace = manufactured_cases["Others"]
    item = next(c for c in load_material_composition().constituents if c.constituent == constituent)
    share = float(item.share)
    gross, incoming_carbon, finished_carbon, losses = reference()
    material = (
        mfa.material_realization.loc[
            mfa.material_realization.year.eq(2019)
            & mfa.material_realization.region.eq("EUR")
            & (
                mfa.material_realization.source_constituent.eq(constituent)
                | mfa.material_realization.destination_constituent.eq(constituent)
            )
        ]
        .groupby("flow_name")
        .amount.sum()
    )
    assert material["manufacturing_principal_input"] == pytest.approx(share * gross, rel=2e-14)
    assert material["manufacturing_finished_output"] == pytest.approx(share, rel=2e-14)
    assert material["manufacturing_principal_loss"] == pytest.approx(
        share * fsum(mass for mass, _ in losses.values()), rel=2e-14
    )
    selected = trace.calculation.loc[
        trace.calculation.region.eq("EUR") & trace.calculation.constituent.eq(constituent)
    ]
    annual = selected.loc[selected.year.eq(2019)]
    actual_losses = annual.loc[annual.custody_status.eq("manufacturing_principal_loss")]
    assert set(actual_losses.source_flow_name) == {f"processing => {name}" for name in losses}
    assert set(actual_losses["product"]) == {"polymer"}
    assert set(actual_losses.unit) == {"Mt C"}
    for name, (_, carbon_loss) in losses.items():
        observed = actual_losses.loc[actual_losses.source_flow_name.eq(f"processing => {name}")]
        actual = observed.set_index("ultimate_carbon_origin").carbon_amount_mt.reindex(_ORIGINS)
        np.testing.assert_allclose(actual, share * carbon_loss * _SHARES, rtol=3e-14, atol=0)
    for year, status in (
        (2019, "manufactured_principal_output"),
        (2019, "entry_into_use"),
        (2020, "incineration_route"),
    ):
        observed = selected.loc[selected.year.eq(year) & selected.custody_status.eq(status)]
        actual = observed.groupby("ultimate_carbon_origin").carbon_amount_mt.sum().reindex(_ORIGINS)
        np.testing.assert_allclose(actual, share * finished_carbon * _SHARES, rtol=3e-14, atol=0)
    assert finished_carbon + fsum(value for _, value in losses.values()) == pytest.approx(
        gross * incoming_carbon, rel=2e-15
    )
    history = next(row for row in trace.history.entries if row[3] == constituent)
    np.testing.assert_allclose(history[-3:], finished_carbon * _SHARES, rtol=3e-14, atol=0)


@pytest.mark.parametrize("family", ("Fibres", "HDPE", "BioPlastics", "PS"))
def test_homogeneous_manufacturing_keeps_carbon_density(
    carbon_authorities, manufactured_cases, family
) -> None:
    """Preserve mixed intrinsic carbon across ordinary forming and fiber principal losses."""
    carbon, _ = carbon_authorities
    mfa, trace = manufactured_cases[family]
    for index, item in enumerate(load_material_composition().constituents):
        if item.material != family:
            continue
        share = float(item.share)
        density = _SHARES @ np.asarray(carbon.fractions[index])
        selected = trace.calculation.loc[
            trace.calculation.region.eq("EUR") & trace.calculation.constituent.eq(item.constituent)
        ]
        for year, status in (
            (2019, "manufactured_principal_output"),
            (2019, "entry_into_use"),
            (2020, "incineration_route"),
        ):
            actual = (
                selected.loc[selected.year.eq(year) & selected.custody_status.eq(status)]
                .groupby("ultimate_carbon_origin")
                .carbon_amount_mt.sum()
                .reindex(_ORIGINS, fill_value=0)
            )
            np.testing.assert_allclose(actual, share * density, rtol=3e-14, atol=0)
        physical = mfa.material_realization.loc[
            mfa.material_realization.year.eq(2019)
            & mfa.material_realization.region.eq("EUR")
            & mfa.material_realization.source_constituent.eq(item.constituent)
            & mfa.material_realization.flow_name.eq("manufacturing_principal_loss")
        ].amount.sum()
        loss = (
            selected.loc[selected.custody_status.eq("manufacturing_principal_loss")]
            .groupby("ultimate_carbon_origin")
            .carbon_amount_mt.sum()
            .reindex(_ORIGINS, fill_value=0)
        )
        # MFA reports G-F; custody sums the admitted individual exits. For
        # tiny PA6 losses their float difference is bounded by precision on
        # gross principal, rather than relative precision on the small loss.
        np.testing.assert_allclose(
            loss, physical * density, rtol=3e-14, atol=np.finfo(float).eps * share
        )


def test_finished_trade_does_not_repeat_selective_manufacturing(carbon_authorities) -> None:
    """Apply UPR retention in its manufacturing region and preserve it through imported use."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Others")
    supplier, consumer = state.regions.index("EUR"), state.regions.index("CHA")
    family, good = state.materials.index("Others"), state.goods.index("Packaging")
    state.annual_inflow[1, consumer, good] = 1
    state.stock_by_cohort[1, 1, consumer, good] = 1
    state.outflow_by_cohort[2, 1, consumer, good] = 1
    inputs.final_exports[1, :, supplier, family, good] = (0.25, 0.75)
    inputs.final_imports[1, :, consumer, family, good] = (0.25, 0.75)
    inputs.daccu_production_rate[1, supplier] = 1
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    selected = trace.calculation.loc[trace.calculation.constituent.eq("thermosets")]
    losses = selected.loc[selected.custody_status.eq("manufacturing_principal_loss")]
    assert set(losses.region) == {"EUR"}
    assert set(losses.year) == {2019}
    item = next(
        c for c in load_material_composition().constituents if c.constituent == "thermosets"
    )
    finished_carbon = _upr_reference()[2]
    for year, status in ((2019, "entry_into_use"), (2020, "incineration_route")):
        rows = selected.loc[selected.year.eq(year) & selected.custody_status.eq(status)]
        assert set(rows.region) == {"CHA"}
        assert set(rows.ultimate_carbon_origin) == {"atmospheric_dac"}
        assert fsum(rows.carbon_amount_mt) == pytest.approx(float(item.share) * finished_carbon)


def test_imported_resin_mixes_before_selective_manufacturing(carbon_authorities) -> None:
    """Mix fossil resin imports with local biogenic resin before UPR's named losses."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Others")
    supplier, consumer = state.regions.index("EUR"), state.regions.index("CHA")
    family, good = state.materials.index("Others"), state.goods.index("Packaging")
    state.annual_inflow[1, consumer, good] = 1
    state.stock_by_cohort[1, 1, consumer, good] = 1
    state.outflow_by_cohort[2, 1, consumer, good] = 1
    inputs.primary_exports[1, :, supplier, family] = (0.125, 0.375)
    inputs.primary_imports[1, :, consumer, family] = (0.125, 0.375)
    inputs.bio_production_rate[1, consumer] = 1
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    selected = trace.calculation.loc[trace.calculation.constituent.eq("thermosets")]
    item = next(
        c for c in load_material_composition().constituents if c.constituent == "thermosets"
    )
    gross, _, finished_carbon, losses = _upr_reference()
    mixture = np.array((0.5 / gross, 1 - 0.5 / gross, 0))
    exits = selected.loc[selected.custody_status.eq("manufacturing_principal_loss")]
    assert set(exits.region) == {"CHA"}
    for name, (_, carbon_loss) in losses.items():
        actual = (
            exits.loc[exits.source_flow_name.eq(f"processing => {name}")]
            .set_index("ultimate_carbon_origin")
            .carbon_amount_mt.reindex(_ORIGINS, fill_value=0)
        )
        np.testing.assert_allclose(actual, float(item.share) * carbon_loss * mixture, rtol=3e-14)
    for year, status in ((2019, "entry_into_use"), (2020, "incineration_route")):
        actual = (
            selected.loc[selected.year.eq(year) & selected.custody_status.eq(status)]
            .set_index("ultimate_carbon_origin")
            .carbon_amount_mt.reindex(_ORIGINS, fill_value=0)
        )
        np.testing.assert_allclose(
            actual, float(item.share) * finished_carbon * mixture, rtol=3e-14
        )


def test_mechanical_recovery_keeps_all_four_loss_fates(carbon_authorities) -> None:
    """Carry each recycling outlet's carbon on its specified loss basis."""
    carbon, recipe = carbon_authorities
    outlets = {
        "processing": "mechanical_resin",
        "waterborne_loss": "mechanical_waterborne_loss",
        "airborne_loss": "mechanical_airborne_loss",
        "landfill": "mechanical_landfilled_reject",
        "incineration": "mechanical_incinerated_reject",
    }
    observed_fates = set()
    for family in ("HDPE", "BioPlastics", "PS"):
        state, inputs = empty_model(family)
        region, good = state.regions.index("EUR"), state.goods.index("Packaging")
        state.annual_inflow[1, region, good] = 1
        state.stock_by_cohort[1, 1, region, good] = 1
        state.outflow_by_cohort[2, 1, region, good] = 1
        state.annual_inflow[2, region, good] = 2
        state.stock_by_cohort[2:, 2, region, good] = 2
        inputs.bio_production_rate[1, region] = 0.25
        inputs.daccu_production_rate[1, region] = 0.25
        inputs.mechanical_recycling_rate[2, region] = 0.4
        inputs.incineration_rate[2, region] = 0.6
        mfa = evaluate_mfa(state, inputs, recipe)
        trace = trace_carbon_custody(mfa, carbon, recipe)
        annual = trace.calculation.loc[
            trace.calculation.year.eq(2020) & trace.calculation.region.eq("EUR")
        ]
        physical = mfa.material_realization.loc[
            mfa.material_realization.year.eq(2020) & mfa.material_realization.region.eq("EUR")
        ]
        for index, item in enumerate(load_material_composition().constituents):
            if item.material != family:
                continue
            density = _SHARES @ np.asarray(carbon.fractions[index])
            material = physical.loc[physical.source_constituent.eq(item.constituent)]
            quantities = material.set_index("flow_name").amount
            selected = annual.loc[annual.constituent.eq(item.constituent)]
            retained_and_lost = np.zeros(3)
            for destination, field in outlets.items():
                actual = (
                    selected.loc[selected.source_flow_name.eq(f"reclmech => {destination}")]
                    .groupby("ultimate_carbon_origin")
                    .carbon_amount_mt.sum()
                    .reindex(_ORIGINS, fill_value=0)
                )
                np.testing.assert_allclose(actual, quantities[field] * density, rtol=3e-14, atol=0)
                retained_and_lost += actual.to_numpy()
                if destination != "processing" and quantities[field] > 0:
                    observed_fates.add(destination)
            np.testing.assert_allclose(
                retained_and_lost, quantities["mechanical_treatment"] * density, rtol=3e-14, atol=0
            )
    assert observed_fates == {"waterborne_loss", "airborne_loss", "landfill", "incineration"}
