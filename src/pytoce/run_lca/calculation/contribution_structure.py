"""Non-additive path and edge attribution for contribution reporting."""

import json
from hashlib import sha256

import numpy as np
import pandas as pd

from .contribution_accumulation import (
    ContributionState,
)
from .contribution_records import (
    _DESCRIPTOR_COLUMNS,
    _BatchedRecords,
)


def _structural_path_id(key: tuple[object, ...]) -> str:
    """Hash the ordered physical, material and provider coordinates of a path."""
    encoded = json.dumps(key, ensure_ascii=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _structural_paths(state: ContributionState) -> pd.DataFrame:
    """Aggregate non-additive process paths for contribution reporting."""
    paths = _BatchedRecords()
    for key, amount in sorted(state.structural_values.items()):
        (
            region,
            year,
            root_database,
            root_code,
            stage_database,
            stage_code,
            life_cycle_stage,
            stage_group,
            cutoff_cycle_role,
            material_cycle_source,
            physical_event_basis,
            physical_event_year,
            source_flow_names,
            lca_fate,
            fate_origin,
            material,
            constituent,
            product,
            virgin_feedstock_route,
            ultimate_carbon_origin,
            good,
            provider_database,
            provider_code,
        ) = key
        path_id = _structural_path_id(key)
        paths.append(
            {
                "path_id": path_id,
                "region": region,
                "year": year,
                "life_cycle_stage": life_cycle_stage,
                "stage_group": stage_group,
                "cutoff_cycle_role": cutoff_cycle_role,
                "material_cycle_source": material_cycle_source,
                "physical_event_basis": physical_event_basis,
                "physical_event_year": physical_event_year,
                "source_flow_names": source_flow_names,
                "lca_fate": lca_fate,
                "fate_origin": fate_origin,
                "material": material,
                "constituent": constituent,
                "product": product,
                "virgin_feedstock_route": virgin_feedstock_route,
                "ultimate_carbon_origin": ultimate_carbon_origin,
                "good": good,
                "root_database": root_database,
                "root_code": root_code,
                "stage_database": stage_database,
                "stage_code": stage_code,
                "provider_database": provider_database,
                "provider_code": provider_code,
                "activation_amount": amount,
                "additivity": "non_additive_explanation",
            }
        )
    return paths.frame()


def _structural_edges(state: ContributionState) -> pd.DataFrame:
    """Aggregate non-additive process edges for contribution reporting."""
    edges = _BatchedRecords()
    for key, amount in sorted(state.structural_values.items()):
        (
            _region,
            _year,
            root_database,
            root_code,
            stage_database,
            stage_code,
            _life_cycle_stage,
            _stage_group,
            _cutoff_cycle_role,
            _material_cycle_source,
            _physical_event_basis,
            _physical_event_year,
            _source_flow_names,
            _lca_fate,
            _fate_origin,
            _material,
            _constituent,
            _product,
            _virgin_feedstock_route,
            _ultimate_carbon_origin,
            _good,
            provider_database,
            provider_code,
        ) = key
        path_id = _structural_path_id(key)
        edges.append(
            {
                "path_id": path_id,
                "edge_step": 1,
                "parent_database": root_database,
                "parent_code": root_code,
                "child_database": stage_database,
                "child_code": stage_code,
                "edge_role": "physical_stage_event",
                "activation_amount": amount,
                "additivity": "non_additive_explanation",
            }
        )
        edges.append(
            {
                "path_id": path_id,
                "edge_step": 2,
                "parent_database": stage_database,
                "parent_code": stage_code,
                "child_database": provider_database,
                "child_code": provider_code,
                "edge_role": "dated_provider_frontier",
                "activation_amount": amount,
                "additivity": "non_additive_explanation",
            }
        )
    return edges.frame()


def _structural_attribution(contributions: pd.DataFrame) -> pd.DataFrame:
    """Format life cycle stage scores for non-additive path and edge tables.

    Args:
        contributions: Regional and global contribution rows, including
            ``life_cycle_stage`` rows indexed by region, requested year, LCIA
            category, and stage.

    Returns:
        Stage-score rows with the columns shared by structural path and edge tables, plus
        one explicit zero residual for each region, year, and LCIA category.

    Notes:
        Structural paths and edges explain relationships between activities. The residual rows keep
        zero-score and
        zero-``contribution_norm`` cases explicit; stage scores retain their calculated values.
    """
    attribution = contributions.loc[
        contributions["contribution_view"].eq("life_cycle_stage")
    ].copy()
    attribution["attribution_role"] = "nonoverlapping_life_cycle_stage"
    denominator_columns = [
        "region",
        "year",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
    ]
    residual = attribution.loc[
        :,
        [*denominator_columns, "share_status"],
    ].drop_duplicates()
    residual["contribution_view"] = "structural_attribution"
    residual["record_role"] = "explicit_residual"
    for column in _DESCRIPTOR_COLUMNS:
        residual[column] = pd.NA if column == "physical_event_year" else ""
    residual["score"] = 0.0
    residual["contribution_norm"] = 0.0
    residual["activation_amount"] = np.nan
    residual["absolute_activation_amount"] = np.nan
    residual["effective_unit_score"] = np.nan
    residual["effective_unit_norm"] = np.nan
    residual["effective_unit_status"] = "not_applicable"
    residual["share"] = np.where(
        residual["share_status"].eq("available"),
        0.0,
        np.nan,
    )
    residual["norm_share"] = np.where(
        residual["share_status"].eq("zero_contribution_norm"),
        np.nan,
        0.0,
    )
    residual["attribution_role"] = "explicit_residual"
    attribution["contribution_view"] = "structural_attribution"
    attribution["physical_event_year"] = attribution["physical_event_year"].astype("Int64")
    residual["physical_event_year"] = residual["physical_event_year"].astype("Int64")
    result = pd.concat((attribution, residual), ignore_index=True)
    return result
