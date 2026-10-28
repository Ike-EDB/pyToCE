"""Compact provider responses and allocated carrying capacity arrays for scoring owners."""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from pytoce.prepare_backgrounds.preparation.premise import ACTIVE_BACKGROUND_SCENARIOS
from pytoce.run_optimization.calculation.candidate_authority import CandidateParentAuthority
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    ProjectionRole,
    ResponseBankProjection,
)
from pytoce.run_optimization.input.allocated_cc_contract import AllocatedCCSelection
from pytoce.run_optimization.input.allocated_cc_contract import (
    _array_identity as allocated_cc_array_identity,
)
from pytoce.run_optimization.input.allocated_cc_product import _selection_identities
from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256
from tests.unit.run_optimization.candidate_characterization_fixtures import _physical_parent

_REGIONS = ("LAM", "OAS", "SSA", "EUR", "NEU", "MEA", "REF", "CAZ", "CHA", "IND", "JPN", "USA")
_RESPONSE_YEARS = tuple(range(2019, 2101))
_BACKGROUNDS = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)


def _response_parent() -> CandidateParentAuthority:
    """Supply one positive kg activation to the response calculation boundary.

    Stock and circularity tables are unused by this score-only owner. This
    fixture cannot initialize an MFA calculation or an intervention planner.
    """
    activations = pd.DataFrame.from_records(
        [
            {
                "year": 2019,
                "region": "EUR",
                "provider_service_id": "provider_fixture",
                "provider_service_code": "provider_fixture_code",
                "demand_type": "virgin_feedstock",
                "stage_group": "production",
                "cutoff_cycle_role": "first_cycle",
                "material_cycle_source": "virgin",
                "physical_event_basis": "production",
                "physical_event_year": 2019,
                "amount": 1.0,
            }
        ]
    )
    return replace(
        _physical_parent(),
        provider_route_identity="provider-routes::fixture",
        activations=activations,
    )


def _allocated_cc_selection(
    *,
    method_compatibility_identity: str = "method-compatibility::fixture",
    climate_compatibility_identity: str = "climate-compatibility::fixture",
) -> AllocatedCCSelection:
    """Return a valid two-case selection in its source region order."""

    regional = np.arange(1, len(_REGIONS) + 1, dtype=np.float64)
    base_annual = np.broadcast_to(regional[:, None, None], (len(_REGIONS), len(_RESPONSE_YEARS), 1))
    annual = np.stack((base_annual, 2.0 * base_annual))
    cumulative = np.stack((regional, 2.0 * regional))
    cases = ("lower_values", "higher_values")
    annual_case_identities = ("annual-lower::fixture", "annual-higher::fixture")
    cumulative_case_identities = (
        "cumulative-lower::fixture",
        "cumulative-higher::fixture",
    )
    table_identities = {
        "annual_table_identity": "annual-table::fixture",
        "cumulative_gwp100_table_identity": "cumulative-table::fixture",
        "method_registry_identity": "method-registry::fixture",
        "cumulative_climate_method_registry_identity": "climate-method::fixture",
        "method_compatibility_identity": method_compatibility_identity,
        "cumulative_climate_compatibility_identity": climate_compatibility_identity,
    }
    identities = _selection_identities(
        result_identity="allocated_cc_result::fixture",
        cases=cases,
        table_identities=table_identities,
    )
    return AllocatedCCSelection(
        allocated_cc_result_identity="allocated_cc_result::fixture",
        optimization_lcia_selection_identity=identities[0],
        allocated_cc_selection_identity=identities[1],
        optimization_lcia_method=("ef31",),
        annual_impact_category_count=1,
        allocated_cc_cases=cases,
        regions=_REGIONS,
        years=_RESPONSE_YEARS,
        annual_categories=(("ef31", "climate_change"),),
        annual_table_identity=table_identities["annual_table_identity"],
        cumulative_gwp100_table_identity=table_identities["cumulative_gwp100_table_identity"],
        method_registry_identity=table_identities["method_registry_identity"],
        cumulative_climate_method_registry_identity=table_identities[
            "cumulative_climate_method_registry_identity"
        ],
        method_compatibility_identity=table_identities["method_compatibility_identity"],
        cumulative_climate_compatibility_identity=table_identities[
            "cumulative_climate_compatibility_identity"
        ],
        annual_allocated_cc_identity=allocated_cc_array_identity(annual),
        cumulative_allocated_cc_identity=allocated_cc_array_identity(cumulative),
        annual_case_table_identities=annual_case_identities,
        cumulative_case_table_identities=cumulative_case_identities,
        annual_allocated_cc=annual,
        cumulative_allocated_cc=cumulative,
    )


def _response_projection(
    role: ProjectionRole, *, unit_score: float = 2.0
) -> ResponseBankProjection:
    """Build a synthetic four-background response projection with one provider anchor."""

    impact_tuple = '["EF v3.1","climate change","global warming potential (GWP100)"]'
    method = {
        "lcia_method": "ef31",
        "selection_order": 0,
        "method_id": "ef31",
        "method_version": "3.1",
        "impact_category_id": "climate_change",
        "impact_category_tuple_json": impact_tuple,
        "unit": "kg CO2-Eq",
        "factor_identity": "f" * 64,
    }
    methods = pd.DataFrame.from_records([method])
    services = pd.DataFrame.from_records(
        [
            {
                "provider_service_id": "provider_fixture",
                "provider_service_code": "provider_fixture_code",
                "provider_unit": "kilogram",
            }
        ]
    )
    signature = {
        key: method[key]
        for key in (
            "method_id",
            "method_version",
            "impact_category_id",
            "impact_category_tuple_json",
            "unit",
            "factor_identity",
        )
    }
    responses = pd.DataFrame.from_records(
        [
            {
                "background_scenario": background,
                "provider_anchor_year": 2020,
                "provider_service_id": "provider_fixture",
                "provider_service_code": "provider_fixture_code",
                "provider_unit": "kilogram",
                **signature,
                "unit_score": unit_score,
                "unit_contribution_norm": abs(unit_score),
            }
            for background in _BACKGROUNDS
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
            for background, year in product(_BACKGROUNDS, _RESPONSE_YEARS)
        ]
    )
    identities = {
        "provider_service_registry_identity": canonical_table_sha256(services),
        "method_registry_identity": canonical_table_sha256(methods),
        "provider_unit_responses_identity": canonical_table_sha256(responses),
        "timex_year_weights_identity": canonical_table_sha256(weights),
        "native_unit_responses_identity": canonical_table_sha256(
            pd.DataFrame(columns=_NATIVE_COLUMNS)
        ),
        "direct_characterization_factors_identity": canonical_table_sha256(
            pd.DataFrame(columns=_DIRECT_COLUMNS)
        ),
    }
    projection_identity = "pytoce_optimization_response_projection_v3::" + (
        payload_sha256(
            {
                "role": role,
                "provider_route_identity": "provider-routes::fixture",
                **identities,
            }
        )
    )
    return ResponseBankProjection(
        role=role,
        response_bank_identity="response-bank::fixture",
        projection_identity=projection_identity,
        provider_route_identity="provider-routes::fixture",
        provider_service_registry_identity=identities["provider_service_registry_identity"],
        provider_service_registry=services,
        method_registry_identity=identities["method_registry_identity"],
        method_registry=methods,
        provider_unit_responses_identity=identities["provider_unit_responses_identity"],
        provider_unit_responses=responses,
        timex_year_weights_identity=identities["timex_year_weights_identity"],
        timex_year_weights=weights,
        native_unit_responses_identity=identities["native_unit_responses_identity"],
        native_unit_responses=pd.DataFrame(columns=_NATIVE_COLUMNS),
        direct_characterization_factors_identity=identities[
            "direct_characterization_factors_identity"
        ],
        direct_characterization_factors=pd.DataFrame(columns=_DIRECT_COLUMNS),
    )
