"""Small PA fiber cohorts for custody and actual-receiving handoff tests."""

from constituent_mfa_fixtures import empty_model, evaluate_mfa

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.run_mfa.model.mfa import MFAEvaluation


def two_cycle_pa_mfa(
    recipe: EcoinventMaterialRecipe, *, biogenic_share: float, dac_share: float
) -> MFAEvaluation:
    """Replace one annual fiber cohort twice, changing new-polymer origins from 2020.

    The fourth year retires the final cohort. Each recycling year collects the
    full preceding cohort, mechanically recycles half, incinerates a quarter
    and landfills the remainder. Real MFA owners allocate the fiber constituents
    and their admitted principal manufacturing/recovery losses. DAC routes start
    in 2020; preceding cohorts therefore retain the fossil reference supply.
    """
    state, inputs = empty_model("Fibres")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    for time in (0, 1, 2):
        state.annual_inflow[time, region, good] = 1
        state.stock_by_cohort[time, time, region, good] = 1
        state.outflow_by_cohort[time + 1, time, region, good] = 1
    for time in (1, 2):
        inputs.mechanical_recycling_rate[time, region] = 0.5
        inputs.incineration_rate[time, region] = 0.25
    for time, year in enumerate(state.years):
        if year >= 2020:
            inputs.bio_production_rate[time, region] = biogenic_share
            inputs.daccu_production_rate[time, region] = dac_share
    return evaluate_mfa(state, inputs, recipe)
