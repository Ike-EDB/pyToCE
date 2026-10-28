"""Select allocated carrying capacity cases for monitoring and target setting."""

import pandas as pd

from .allocated_cc_contract import (
    _YEARS,
    AllocatedCCError,
    AllocatedCCResult,
    AllocatedCCSelection,
    _array_identity,
    _h12_regions,
)
from .allocated_cc_product import (
    _compile_selection_arrays,
    _selection_identities,
    _selection_table_identities,
)


def select_allocated_cc_cases(
    result: AllocatedCCResult,
    allocated_cc_case: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Admit normalized case names and retain the completed result's case order."""

    if allocated_cc_case is None:
        return result.allocated_cc_cases
    unknown = tuple(case for case in allocated_cc_case if case not in result.allocated_cc_cases)
    if unknown:
        raise AllocatedCCError(
            f"The allocated carrying capacity (aCC) result "
            f"{result.allocated_cc_result!r} does not contain allocated_cc_case={unknown!r}. "
            f"Choose from {result.allocated_cc_cases!r}, or pass None to use every available "
            "case."
        )
    requested = set(allocated_cc_case)
    return tuple(case for case in result.allocated_cc_cases if case in requested)


def select_allocated_cc_result(
    result: AllocatedCCResult,
    cases: tuple[str, ...],
) -> AllocatedCCSelection:
    """Load the admitted aCC cases and tables used to calculate ASR.

    In optimization, the ``lcia_method`` used for final results is a separate
    workflow input.
    """

    directory = result.result_directory
    annual = pd.read_parquet(directory / "annual_allocated_cc.parquet")
    methods = pd.read_parquet(directory / "lcia_method_registry.parquet")
    cumulative_method = pd.read_parquet(directory / "cumulative_climate_method_registry.parquet")
    cumulative = pd.read_parquet(directory / "cumulative_gwp100_allocated_cc.parquet")
    annual = annual.loc[annual["allocated_cc_case"].isin(cases)].reset_index(drop=True)
    cumulative = cumulative.loc[cumulative["allocated_cc_case"].isin(cases)].reset_index(drop=True)
    table_identities = _selection_table_identities(annual, cumulative, methods, cumulative_method)
    (
        annual_values,
        cumulative_values,
        annual_categories,
        annual_case_identities,
        cumulative_case_identities,
    ) = _compile_selection_arrays(annual, cumulative, methods, cases)
    annual_array_identity = _array_identity(annual_values)
    cumulative_array_identity = _array_identity(cumulative_values)
    requested_lcia_identity, selection_identity = _selection_identities(
        result_identity=result.allocated_cc_result_identity,
        cases=cases,
        table_identities=table_identities,
    )
    return AllocatedCCSelection(
        allocated_cc_result_identity=result.allocated_cc_result_identity,
        allocated_cc_selection_identity=selection_identity,
        optimization_lcia_selection_identity=requested_lcia_identity,
        optimization_lcia_method=("ef31",),
        annual_impact_category_count=len(methods),
        allocated_cc_cases=cases,
        regions=_h12_regions(),
        years=_YEARS,
        annual_categories=annual_categories,
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
        annual_allocated_cc_identity=annual_array_identity,
        cumulative_allocated_cc_identity=cumulative_array_identity,
        annual_case_table_identities=annual_case_identities,
        cumulative_case_table_identities=cumulative_case_identities,
        annual_allocated_cc=annual_values,
        cumulative_allocated_cc=cumulative_values,
    )
