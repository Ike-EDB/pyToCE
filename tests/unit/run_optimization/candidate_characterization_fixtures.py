"""Selected-candidate characterization and material-carbon custody coverage."""

from itertools import product

import numpy as np
import pandas as pd

from pytoce.prepare_backgrounds.preparation.premise import ACTIVE_BACKGROUND_SCENARIOS
from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_lca.calculation.providers import PreparedProviderProduct
from pytoce.run_mfa.model.optimization_parents import (
    OPTIMIZATION_CIRCULARITY_PARENT_METHOD_IDENTITY,
    OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY,
    OptimizationCircularityParent,
    OptimizationDemandLifetimeParent,
)
from pytoce.run_optimization.calculation.candidate_authority import (
    CandidateParentAuthority,
    CandidateParentMember,
    PhysicalEvaluation,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    CandidateParentScoreProjection,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import (
    PaCandidateProjection,
    PaCandidateScores,
    build_pa_candidate_projection,
    calculate_pa_candidate_projection,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    ResponseBankProjection,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256
from tests.pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources

_BACKGROUNDS = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
_YEARS = tuple(range(2019, 2101))


def _empty_pa_scoring(
    methods: pd.DataFrame, *, backgrounds: tuple[str, ...] = _BACKGROUNDS
) -> tuple[PaCandidateProjection, PaCandidateScores]:
    """Calculate the actual empty PA stream from complete detached source records.

    This score-only test requests zero PA receiving events. Complete typed member
    records supply the source handoff with an empty receiving inventory.
    """
    projection = _result_projection(methods)
    reference = pa_receiving_reference()
    providers = {
        background: PreparedProviderProduct(
            database_by_year={2025: "source"},
            timeline_database_by_year={2025: "temporal-source"},
            pa_receiving_reference=reference,
            pa_receiving_sources_by_year={
                2025: pa_receiving_sources(background_scenario=background)
            },
        )
        for background in backgrounds
    }
    context = build_pa_candidate_projection(
        projection,
        providers,
        {site.infiltration_class: 0.2 for site in reference.dump.sites},
        prepared_receiving_identity="prepared-receiving::fixture",
    )
    empty = calculate_pa_candidate_projection(
        context, spinning_events=(), recycling_events=(), disposal_events=()
    )
    return context, empty


def _pa_scores_at_2050(
    methods: pd.DataFrame,
    *,
    scores: tuple[float, ...],
    norms: tuple[float, ...],
    backgrounds: tuple[str, ...] = _BACKGROUNDS,
) -> PaCandidateScores:
    """Place coherent deterministic full-method quantities at EUR/2050.

    These signed scores and nonnegative contribution totals test contraction
    and parent subtraction only. Actual native receiving equivalence belongs
    to the independent projection test. Matching stage rows retain the same
    full method order, units and annual quantities.
    """
    shape = (len(backgrounds), len(H12_REGIONS), len(_YEARS), len(methods))
    annual = np.zeros(shape, dtype=np.float64)
    absolute = np.zeros(shape, dtype=np.float64)
    annual[:, H12_REGIONS.index("EUR"), _YEARS.index(2050), :] = scores
    absolute[:, H12_REGIONS.index("EUR"), _YEARS.index(2050), :] = norms
    rows = [
        {
            "characterization_background_scenario": background,
            "region": "EUR",
            "year": 2050,
            **{
                key: method[key]
                for key in (
                    "method_id",
                    "method_version",
                    "impact_category_id",
                    "impact_category_tuple_json",
                    "unit",
                    "factor_identity",
                )
            },
            "life_cycle_stage": "incineration_treatment",
            "stage_group": "managed_terminal_end_of_life",
            "cutoff_cycle_role": "waste_producer_terminal_treatment",
            "material_cycle_source": "not_applicable",
            "physical_event_basis": "year collected end-of-life waste is incinerated",
            "physical_event_year": 2050,
            "score": score,
            "contribution_norm": norm,
        }
        for background in backgrounds
        for method, score, norm in zip(methods.to_dict("records"), scores, norms, strict=True)
    ]
    stages = pd.DataFrame.from_records(rows)
    stages = stages.astype(
        {
            column: "int64"
            if column in ("year", "physical_event_year")
            else "float64"
            if column in ("score", "contribution_norm")
            else "string"
            for column in stages
        }
    )
    annual.setflags(write=False)
    absolute.setflags(write=False)
    return PaCandidateScores(annual, absolute, stages)


def _method_registry() -> pd.DataFrame:
    """Return the one exact reporting method used by the compact fixture."""

    return pd.DataFrame.from_records(
        [
            {
                "lcia_method": "ef31",
                "selection_order": 0,
                "method_id": "ef31",
                "method_version": "3.1",
                "impact_category_id": "climate_change",
                "impact_category_tuple_json": (
                    '["EF v3.1","climate change","global warming potential (GWP100)"]'
                ),
                "unit": "kg CO2-Eq",
                "factor_identity": "f" * 64,
            }
        ]
    )


def _physical_parent() -> CandidateParentAuthority:
    """Build the exact typed parent MFA result needed by selected characterization."""

    demand_parent = OptimizationDemandLifetimeParent(
        identity="demand-parent::fixture",
        method_identity=OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY,
        parent_foreground_scenario_identity="scenario::fixture",
        physical_state_identity="physical-state::fixture",
        stock_target_identity="stock-target::fixture",
        lifetime_identity="lifetime::fixture",
        stock_targets=pd.DataFrame(),
        lifetimes=pd.DataFrame(),
    )
    circularity_parent = OptimizationCircularityParent(
        identity="circularity-parent::fixture",
        method_identity=OPTIMIZATION_CIRCULARITY_PARENT_METHOD_IDENTITY,
        physical_state_identity="physical-state::fixture",
        foreground_scenario="frozen_2019_default_baseline",
        table=pd.DataFrame(),
    )
    return CandidateParentAuthority(
        identity="physical-parent::fixture",
        foreground_scenario_identity="scenario::fixture",
        mfa_result_identity="mfa::fixture",
        physical_state_identity="physical-state::fixture",
        provider_route_identity="routes::fixture",
        activations=pd.DataFrame(),
        pa_spinning_events=(),
        pa_recycling_events=(),
        pa_disposal_events=(),
        demand_lifetime_parent=demand_parent,
        circularity_parent=circularity_parent,
    )


def _result_projection(methods: pd.DataFrame) -> ResponseBankProjection:
    """Build synthetic annual one-service responses using a single provider anchor."""

    service = pd.DataFrame.from_records(
        [
            {
                "provider_service_id": "service",
                "provider_service_code": "service-code",
                "provider_unit": "kilogram",
            }
        ]
    )
    signatures = methods.loc[
        :,
        [
            "method_id",
            "method_version",
            "impact_category_id",
            "impact_category_tuple_json",
            "unit",
            "factor_identity",
        ],
    ].to_dict("records")
    responses = pd.DataFrame.from_records(
        [
            {
                "background_scenario": background,
                "provider_anchor_year": 2020,
                "provider_service_id": "service",
                "provider_service_code": "service-code",
                "provider_unit": "kilogram",
                **signature,
                "unit_score": 2.0,
                "unit_contribution_norm": 3.0,
            }
            for background, signature in product(_BACKGROUNDS, signatures)
        ]
    )
    weights = pd.DataFrame.from_records(
        [
            {
                "background_scenario": background,
                "occurrence_year": year,
                "provider_anchor_year": 2020,
                "temporal_provider_share": 1.0,
            }
            for background, year in product(_BACKGROUNDS, _YEARS)
        ]
    )
    return ResponseBankProjection(
        role="result",
        response_bank_identity="response-bank::fixture",
        projection_identity="result-projection::fixture",
        provider_route_identity="routes::fixture",
        provider_service_registry_identity=canonical_table_sha256(service),
        provider_service_registry=service,
        method_registry_identity=canonical_table_sha256(methods),
        method_registry=methods,
        provider_unit_responses_identity=canonical_table_sha256(responses),
        provider_unit_responses=responses,
        timex_year_weights_identity=canonical_table_sha256(weights),
        timex_year_weights=weights,
        native_unit_responses_identity="no-pa-native::fixture",
        native_unit_responses=pd.DataFrame(columns=_NATIVE_COLUMNS),
        direct_characterization_factors_identity="no-pa-direct::fixture",
        direct_characterization_factors=pd.DataFrame(columns=_DIRECT_COLUMNS),
    )


def _score_parent(
    methods: pd.DataFrame, physical_parent: CandidateParentAuthority
) -> CandidateParentScoreProjection:
    """Score 3 kg per region/year at 2 impact units and 3 absolute units per kg."""

    signature = (
        methods.iloc[0]
        .loc[
            [
                "method_id",
                "method_version",
                "impact_category_id",
                "impact_category_tuple_json",
                "unit",
                "factor_identity",
            ],
        ]
        .to_dict()
    )
    annual = pd.DataFrame.from_records(
        [
            {
                "background_scenario": background,
                "region": region,
                "year": year,
                **signature,
                "score": 6.0,
                "contribution_norm": 9.0,
            }
            for background, region, year in product(_BACKGROUNDS, H12_REGIONS, _YEARS)
        ]
    )
    members = tuple(
        CandidateParentMember(
            background_scenario=background,
            parent_score_identity=f"parent::{background}",
            projected_annual_scores_identity=f"annual::{background}",
        )
        for background in _BACKGROUNDS
    )
    return CandidateParentScoreProjection(
        role="result",
        identity="result-parent::fixture",
        physical_parent_identity=physical_parent.identity,
        annual_scores_identity=canonical_table_sha256(annual),
        method_registry_identity=canonical_table_sha256(methods),
        member_parents=members,
        annual_scores=annual,
        method_registry=methods,
    )


def _physical(identity: str, *, delta: pd.DataFrame) -> PhysicalEvaluation:
    """Return a selected MFA candidate fixture with the supplied activation delta."""

    return PhysicalEvaluation(
        physical_request_identity=f"request::{identity}",
        physical_candidate_identity=f"candidate::{identity}",
        physical_parent_identity="physical-parent::fixture",
        intervention_authority_identity=f"intervention::{identity}",
        status="physical_feasible",
        diagnostic=None,
        physical_state_identity=f"physical-state::{identity}",
        activation_identity=f"activation::{identity}",
        activation_delta=delta,
        pa_spinning_events=(),
        pa_recycling_events=(),
        pa_disposal_events=(),
    )
