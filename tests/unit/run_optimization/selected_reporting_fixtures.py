"""Focused tests for selected-only global and life-cycle-stage reporting."""

from collections.abc import Sequence
from dataclasses import replace
from math import fsum
from pathlib import Path
from typing import cast

import pandas as pd
from candidate_worker_fixtures import _summary

from pytoce.prepare_backgrounds.preparation.premise import (
    ACTIVE_BACKGROUND_SCENARIOS,
)
from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.calculation.pa_candidate_projection import PaCandidateScores
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    ResponseBankProjection,
)
from pytoce.run_optimization.calculation.selected_reporting import (
    BaselineLCAResult,
    BaselineLCAResultRecord,
    SelectedReportingState,
)
from pytoce.run_optimization.calculation.selected_stage_reporting import (
    stream_selected_stage_reporting,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    SelectedCandidateResult,
    _selected_candidate_result,
    decode_candidate_worker_result,
    encode_candidate_worker_result,
)
from pytoce.run_optimization.runtime.result_store import (
    stage_selected_life_cycle_scores,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256

BACKGROUNDS = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
METHOD = {
    "method_id": "ef31",
    "method_version": "3.1",
    "impact_category_id": "climate_change",
    "impact_category_tuple_json": (
        '["EF v3.1","climate change","global warming potential (GWP100)"]'
    ),
    "unit": "kg CO2-Eq",
    "factor_identity": "a" * 64,
}
STAGE = {
    "life_cycle_stage": "production",
    "stage_group": "material_production",
    "cutoff_cycle_role": "primary_material_cycle",
    "material_cycle_source": "primary_resin",
    "physical_event_basis": "year material enters system",
}


def _projection(
    unit_scores: tuple[float, ...],
    *,
    unit_contribution_norm: float = 3.0,
    backgrounds: tuple[str, ...] = BACKGROUNDS,
) -> ResponseBankProjection:
    """Build synthetic per-background scores with one 2030 provider anchor."""
    registry = pd.DataFrame.from_records([{"lcia_method": "ef31", "selection_order": 0, **METHOD}])
    services = pd.DataFrame.from_records(
        [
            {
                "provider_service_id": "service::one",
                "provider_service_code": "service_one",
                "provider_unit": "kilogram",
            }
        ]
    )
    responses = pd.DataFrame.from_records(
        [
            {
                "background_scenario": background,
                "provider_anchor_year": 2030,
                "provider_service_id": "service::one",
                "provider_service_code": "service_one",
                "provider_unit": "kilogram",
                **METHOD,
                "unit_score": unit_score,
                "unit_contribution_norm": unit_contribution_norm,
            }
            for background, unit_score in zip(backgrounds, unit_scores, strict=True)
        ]
    )
    weights = pd.DataFrame.from_records(
        [
            {
                "background_scenario": background,
                "occurrence_year": year,
                "provider_anchor_year": 2030,
                "temporal_provider_share": 1.0,
            }
            for background in backgrounds
            for year in range(2019, 2101)
        ]
    )
    return ResponseBankProjection(
        role="result",
        response_bank_identity="response-bank::fixture",
        projection_identity="result-projection::fixture",
        provider_route_identity="provider-route::fixture",
        provider_service_registry_identity=canonical_table_sha256(services),
        provider_service_registry=services,
        method_registry_identity=canonical_table_sha256(registry),
        method_registry=registry,
        provider_unit_responses_identity=canonical_table_sha256(responses),
        provider_unit_responses=responses,
        timex_year_weights_identity=canonical_table_sha256(weights),
        timex_year_weights=weights,
        native_unit_responses_identity="no-pa-native::fixture",
        native_unit_responses=pd.DataFrame(columns=_NATIVE_COLUMNS),
        direct_characterization_factors_identity="no-pa-direct::fixture",
        direct_characterization_factors=pd.DataFrame(columns=_DIRECT_COLUMNS),
    )


def _baseline_lca_result(
    background: str,
    *,
    parent_score: float,
    parent_norm: float = 3.0,
) -> BaselineLCAResult:
    annual_rows = []
    stage_rows = []
    for region in H12_REGIONS:
        for year in range(2019, 2101):
            changed = region == "EUR" and year == 2020
            combined_score = (
                {
                    "CAZ": 1.0e16,
                    "CHA": 1.0,
                    "EUR": -1.0e16,
                }.get(region, 0.0)
                if year == 2021
                else 0.0
            )
            score = parent_score if changed else combined_score
            norm = parent_norm if changed else abs(combined_score)
            annual_rows.append(
                {
                    "region": region,
                    "year": year,
                    **{column: METHOD[column] for column in tuple(METHOD)[:-1]},
                    "score": score,
                    "contribution_norm": norm,
                }
            )
            stage_rows.append(
                {
                    **annual_rows[-1],
                    **STAGE,
                    "physical_event_year": year,
                }
            )
    annual_stage = pd.DataFrame.from_records(stage_rows)
    cumulative = (
        annual_stage.groupby(
            ["region", *tuple(METHOD)[:-1], *tuple(STAGE)],
            observed=True,
            sort=True,
        )[["score", "contribution_norm"]]
        .sum()
        .reset_index()
    )
    cumulative["year"] = pd.NA
    cumulative["physical_event_year"] = pd.NA
    record = BaselineLCAResultRecord(
        foreground_scenario="frozen_2019_default_baseline",
        background_scenario=background,
        result_identity=f"baseline-lca-result::{background}",
        result_manifest_sha256="b" * 64,
        annual_life_cycle_stage_scores_identity=canonical_table_sha256(annual_stage),
        life_cycle_stage_scores_identity=canonical_table_sha256(cumulative),
    )
    return BaselineLCAResult(
        record=record,
        annual_life_cycle_stage_scores=annual_stage,
        life_cycle_stage_scores=cumulative,
    )


def _states(
    background: str,
    *,
    physical_candidate_identity: str = "physical-candidate::fixture",
) -> tuple[SelectedReportingState, ...]:
    return (
        SelectedReportingState(
            physical_candidate_identity=physical_candidate_identity,
            state_identity="selection::environmental_minimum",
            selection_candidate_identity="candidate::2050",
            timing_2100_candidate_identity="candidate::2100",
            timing_2100_physical_candidate_identity=physical_candidate_identity,
            optimization_block="demand_lifetime_frontier",
            selection_background_scenario=background,
            allocated_cc_case="p50",
            selection_status="intervention_compliant",
            all_region_joint_compliant=True,
            intervention_distance=0.25,
            binding_slack=1.0 - 0.9,
            compliant_region_count=12,
            fully_compliant_impact_target_count=17,
            compliant_region_impact_group_count=204,
            compliant_coordinate_count=15_564,
            region_count=12,
            impact_target_count=17,
            region_impact_group_count=204,
            coordinate_count=15_564,
            worst_asr=0.9,
            maximum_positive_exceedance=0.0,
            balanced_mean_positive_exceedance=0.0,
            fully_compliant_regions=H12_REGIONS,
            fully_compliant_impact_targets=tuple(
                f"annual::ef31::category-{index}" for index in range(16)
            )
            + ("cumulative_gwp100::ef31::climate_change",),
            binding_coordinates=(("EUR", "annual", "ef31", "climate_change", 2050, 0.9),),
            transgressing_coordinates=(),
            selected_is_exact_parent=False,
            selected_has_non_grid_baseline_component=False,
            declared_neighborhood_verified=True,
            selection_stable=True,
            demand_reduction_pct=25,
            lifetime_extension_pct=25,
            collection_pct=10,
            mechanical_recycling_pct=40,
            chemical_recycling_pct=20,
            incineration_pct=30,
            landfill_pct=10,
            fossil_feedstock_pct=70,
            biogenic_feedstock_pct=20,
            daccu_feedstock_pct=10,
            selection_target_year=2050,
        ),
    )


def _candidate_evaluation(
    *,
    candidate_amount: float,
) -> SelectedCandidateResult:
    from candidate_worker_fixtures import _result_and_acc_metrics

    delta = pd.DataFrame.from_records(
        [
            {
                "year": 2020,
                "region": "EUR",
                "provider_service_id": "service::one",
                "provider_service_code": "service_one",
                **STAGE,
                "physical_event_year": 2020,
                "parent_amount_kg": 1.0,
                "candidate_amount_kg": candidate_amount,
                "delta_amount_kg": candidate_amount - 1.0,
            }
        ]
    )
    result = decode_candidate_worker_result(
        encode_candidate_worker_result(_result_and_acc_metrics(activation_delta=delta)[0])
    )
    summary = _summary(result)
    selected = _selected_candidate_result(
        result,
        summary,
        selection_identities=("selection::environmental_minimum",),
    )
    return selected


def _parents(unit_scores: tuple[float, ...]) -> tuple[BaselineLCAResult, ...]:
    return tuple(
        _baseline_lca_result(background, parent_score=unit_score)
        for background, unit_score in zip(BACKGROUNDS, unit_scores, strict=True)
    )


def _with_pa_parent(parent: BaselineLCAResult, pa_parent: PaCandidateScores) -> BaselineLCAResult:
    """Add actual PA stage scores to a completed synthetic fixed-service parent.

    Both quantities have already been characterized in the same method units.
    Their annual and cumulative products retain all stage coordinates; the
    published parent contains their sum before selected endpoint subtraction.
    """
    pa_rows = pa_parent.stage_scores.loc[
        pa_parent.stage_scores.characterization_background_scenario.eq(
            parent.record.background_scenario
        )
    ].drop(columns=["characterization_background_scenario", "factor_identity"])
    keys = ["region", "year", *tuple(METHOD)[:-1], *STAGE, "physical_event_year"]
    annual = (
        pd.concat((parent.annual_life_cycle_stage_scores, pa_rows), ignore_index=True)
        .groupby(keys, observed=True, sort=True)[["score", "contribution_norm"]]
        .agg(fsum)
        .reset_index()
    )
    cumulative = (
        annual.groupby(
            [key for key in keys if key not in ("year", "physical_event_year")],
            observed=True,
            sort=True,
        )[["score", "contribution_norm"]]
        .agg(fsum)
        .reset_index()
    )
    cumulative["year"] = pd.NA
    cumulative["physical_event_year"] = pd.NA
    return replace(
        parent,
        record=replace(
            parent.record,
            annual_life_cycle_stage_scores_identity=canonical_table_sha256(annual),
            life_cycle_stage_scores_identity=canonical_table_sha256(cumulative),
        ),
        annual_life_cycle_stage_scores=annual,
        life_cycle_stage_scores=cumulative,
    )


def _stream_stage(
    tmp_path: Path,
    *,
    selected_evaluations: Sequence[SelectedCandidateResult],
    states: Sequence[SelectedReportingState],
    baseline_lca_results: Sequence[BaselineLCAResult],
    response_projection: ResponseBankProjection,
    pa_parent: PaCandidateScores,
) -> tuple[pd.DataFrame, str]:
    """Round-trip selected fixed and actual PA stages through the real result writer."""
    directory = tmp_path / "selected"
    with stage_selected_life_cycle_scores(directory) as writer:
        stream_selected_stage_reporting(
            selected_evaluations=selected_evaluations,
            states=states,
            baseline_lca_results=baseline_lca_results,
            response_projection=response_projection,
            pa_parent=pa_parent,
            pa_candidates={
                item.result.physical.physical_candidate_identity: cast(
                    PaCandidateScores, item.result.pa_scores
                )
                for item in selected_evaluations
            },
            sink=writer,
            progress=lambda _completed, _total: None,
        )
        staged = writer.finish()
        stage = pd.read_parquet(
            staged.staging_directory / "selected_life_cycle_stage_scores.parquet"
        )
    return stage, staged.canonical_sha256
