"""Coordinate the single direct MFA calculation used by public and candidate runs."""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256

import numpy as np
import pandas as pd

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.shared.runtime.filesystem import (
    canonical_json_bytes,
    canonical_table_sha256,
    payload_sha256,
)

from ..remind_mfa.execution import (
    REMIND_MFA_ADAPTER_METHOD,
    RemindMFAExecutionRequest,
    RemindMFAExecutionSession,
)
from ..remind_mfa.process_inputs import (
    SOURCE_TRADE_BALANCE_INFEASIBLE,
    RemindMFAProcessInputs,
)
from ..remind_mfa.translation import translate_annual_inputs
from ..scenario.catalogue import ParameterDefinition
from ..scenario.rules import OPTIMIZATION_PARENT_OVERLAY_ORIGIN
from .cohorts import (
    FixedLifetimeStockSystem,
    ProspectiveCohortResult,
    accepted_default_cohort_result,
    continue_cohort_stock,
    fixed_lifetime_stock_system,
)
from .flows import (
    DIRECT_MFA_METHOD,
    MFA_FLOW_FEASIBILITY_CODES,
    NOMINAL_FEEDSTOCK_METHOD,
    MFAFlowState,
    calculate_mfa_flows,
)
from .materials import (
    MATERIAL_FEASIBILITY_CODES,
    MATERIAL_REALIZATION_METHOD,
    MaterialComposition,
    MaterialRealization,
    material_realization_table,
)
from .result_tables import (
    _cohort_composition_table,
    _cohort_material_identity_table,
    _cohort_state_table,
    _derived_outputs,
    _optimization_cohort_material_identity_table,
    _optimization_cohort_state_table,
    _resolved_input_view,
    mfa_flow_table,
    nominal_feedstock_table,
)
from .sankey import build_mfa_sankey
from .stocks import ACCUMULATED_STOCKS, MFAStockState, accumulate_mfa_stocks, mfa_stock_table

OPTIMIZATION_CANDIDATE_MFA_METHOD_IDENTITY = DIRECT_MFA_METHOD
OPTIMIZATION_MFA_FEASIBILITY_CODES = (
    MFA_FLOW_FEASIBILITY_CODES | MATERIAL_FEASIBILITY_CODES | {SOURCE_TRADE_BALANCE_INFEASIBLE}
)
MFA_PHYSICAL_STATE_SCHEMA = "pytoce_mfa_physical_state_schema_v1"
_PROSPECTIVE_CHANGE_ORIGINS = frozenset(
    {"user_foreground_scenario", OPTIMIZATION_PARENT_OVERLAY_ORIGIN}
)
_OPTIMIZATION_COHORT_CACHE_SIZE = 2


@dataclass(frozen=True, slots=True)
class MFAEvaluation:
    """Admitted baseline tables required by optimization and foreground construction."""

    model_flows: pd.DataFrame
    derived_outputs: pd.DataFrame
    material_realization: pd.DataFrame
    nominal_feedstock_equivalents: pd.DataFrame
    cohort_state: pd.DataFrame
    cohort_material_identity: pd.DataFrame
    physical_state_identity: str
    material_identity_method: str


@dataclass(frozen=True, slots=True)
class MFACandidateEvaluation:
    """The validated candidate MFA inputs needed by foreground construction."""

    model_flows: pd.DataFrame
    material_realization: pd.DataFrame
    nominal_feedstock_equivalents: pd.DataFrame
    cohort_state: pd.DataFrame
    cohort_material_identity: pd.DataFrame
    physical_state_identity: str
    material_identity_method: str


@dataclass(frozen=True, slots=True)
class MFAComputation:
    """Complete direct MFA tables ready for their one manifest-last publication."""

    model_flows: pd.DataFrame
    model_stocks: pd.DataFrame
    material_realization: pd.DataFrame
    nominal_feedstock_equivalents: pd.DataFrame
    reconciliation_ledger: pd.DataFrame
    stock_balance: pd.DataFrame
    transition_ledger: pd.DataFrame
    waste_trade_propensity_ledger: pd.DataFrame
    resolved_inputs: pd.DataFrame
    derived_outputs: pd.DataFrame
    cohort_state: pd.DataFrame
    cohort_material_identity: pd.DataFrame
    cohort_composition: pd.DataFrame
    sankey_edges: pd.DataFrame
    sankey_nodes: pd.DataFrame
    physical_state_identity: str
    material_identity_method: str


@dataclass(frozen=True, slots=True)
class _HistoricalMFAState:
    """Immutable 1950--2019 prefix and its candidate reporting boundary."""

    flows: MFAFlowState
    reporting_boundary: MFAFlowState
    flow_signature: str
    nominal_signature: str
    material_signature: str


@dataclass(frozen=True, slots=True)
class _ValidatedMFAState:
    """Prospective MFA arrays and input provenance before ordinary report expansion."""

    cohort: ProspectiveCohortResult
    cohort_identity: str
    future: MFAFlowState
    history: _HistoricalMFAState
    waste_trade_propensity_ledger: pd.DataFrame
    physical_state_identity: str


def _mfa_array_signature(
    coordinates: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> str:
    """Identify named MFA arrays using declared axes and little-endian float64.

    Values retain their exact bits and C-order coordinates. This identifies the
    MFA calculation from its calculated arrays; written Parquet
    bytes and public table schemas have their own artifact identities.
    """
    digest = sha256(
        canonical_json_bytes(
            {
                "schema_version": "pytoce_mfa_array_authority_schema_v1",
                "coordinates": dict(coordinates),
            }
        )
    )
    for name, raw in sorted(arrays.items()):
        values = np.ascontiguousarray(raw, dtype="<f8")
        digest.update(canonical_json_bytes({"name": name, "shape": values.shape, "dtype": "<f8"}))
        digest.update(values.data.cast("B"))
    return digest.hexdigest()


def _cohort_signature(cohort: ProspectiveCohortResult) -> str:
    """Bind in-use cohort stock, movements and material identity in Mt and fractions."""
    return _mfa_array_signature(
        {
            "year": cohort.years,
            "region": cohort.regions,
            "material": cohort.materials,
            "good": cohort.goods,
            "element": cohort.elements,
            "material_identity_method": cohort.material_identity_method,
        },
        {
            "stock_by_cohort_mt": cohort.stock_by_cohort,
            "outflow_by_cohort_mt": cohort.outflow_by_cohort,
            "annual_inflow_mt": cohort.annual_inflow,
            "material_share": cohort.cohort_material_shares,
            "nominal_element_fraction": cohort.carbon_content,
        },
    )


def _flow_signatures(state: MFAFlowState) -> tuple[str, str]:
    """Bind polymer movements and the separate nominal feedstock account."""
    coordinates = {
        "year": state.years,
        "region": state.regions,
        "material": state.materials,
        "good": state.goods,
    }
    return (
        _mfa_array_signature({**coordinates, "unit": "Mt polymer"}, state.flows),
        _mfa_array_signature(
            {**coordinates, "unit": "Mt polymer equivalent", "method": NOMINAL_FEEDSTOCK_METHOD},
            state.nominal_feedstock_equivalents,
        ),
    )


def _material_signature(states: tuple[MaterialRealization, ...]) -> str:
    """Bind constituent amounts and gross products."""
    signatures = []
    for state in states:
        arrays = {
            "primary_imports": state.primary_imports,
            "primary_exports": state.primary_exports,
        }
        for name, quantities in (
            ("polymer", state.amounts),
            ("hydrothermal_product", state.hydrothermal_products),
            ("cracker_product", state.cracker_products),
            ("recovered_monomer", state.recovered_monomers),
            ("virgin_monomer", state.virgin_monomers),
        ):
            arrays.update(
                {name + ":" + quantity: values for quantity, values in quantities.items()}
            )
        signatures.append(
            _mfa_array_signature(
                {
                    "year": state.year,
                    "region": state.region,
                    "unit": "Mt",
                    # The session already binds composition and portfolio sources.
                    # These labels identify the array axis
                    # for every regional year and candidate.
                    "constituents": [
                        (item.material, item.constituent) for item in state.constituents
                    ],
                    "family_index": state.family_index.tolist(),
                },
                arrays,
            )
        )
    return payload_sha256(signatures)


def _changed_parameter_ids(compiled: pd.DataFrame) -> frozenset[str]:
    """Identify authored prospective controls, including explicit default rules."""
    return frozenset(
        compiled.loc[
            compiled["value_origin"].isin(_PROSPECTIVE_CHANGE_ORIGINS), "parameter_id"
        ].astype(str)
    )


def _first_changed_year(compiled: pd.DataFrame, parameter_ids: set[str]) -> int:
    """Find the first changed year for a known nonempty set of changed controls."""
    return int(
        compiled.loc[
            compiled["parameter_id"].isin(parameter_ids)
            & compiled["value_origin"].isin(_PROSPECTIVE_CHANGE_ORIGINS),
            "year",
        ].min()
    )


def _require_validity(checks: pd.DataFrame) -> None:
    """Report a failed calculated mass balance."""
    invalid = checks.loc[~checks["status"].astype(str).str.startswith("ok")]
    if not invalid.empty:
        raise ValueError(f"Direct MFA balance checks failed: {invalid.to_dict('records')}")


def mfa_evaluation_session_identity(
    request: RemindMFAExecutionRequest,
    parameter_catalogue: tuple[ParameterDefinition, ...],
    *,
    material_recipe_identity: str,
    material_composition_identity: str,
) -> str:
    """Bind one session to the exact admitted source, controls and material authorities."""
    return mfa_evaluation_session_identity_from_source(
        request.identity.as_dict(),
        parameter_catalogue,
        material_recipe_identity=material_recipe_identity,
        material_composition_identity=material_composition_identity,
    )


def mfa_evaluation_session_identity_from_source(
    remind_mfa_identity: Mapping[str, str],
    parameter_catalogue: tuple[ParameterDefinition, ...],
    *,
    material_recipe_identity: str,
    material_composition_identity: str,
) -> str:
    """Bind current calculation methods to the admitted source and material inputs.

    A saved parent can record an earlier adapter. Candidate planning and worker
    execution both use the current adapter; the parent's original method stays
    in its separate result provenance.
    """
    return "pytoce_mfa_session_schema_v1::" + payload_sha256(
        {
            "method_identity": DIRECT_MFA_METHOD,
            "material_realization_method": MATERIAL_REALIZATION_METHOD,
            "nominal_feedstock_method": NOMINAL_FEEDSTOCK_METHOD,
            "remind_mfa_identity": {
                **remind_mfa_identity,
                "adapter_identity": REMIND_MFA_ADAPTER_METHOD,
            },
            "parameter_catalogue": [asdict(item) for item in parameter_catalogue],
            "material_recipe_identity": material_recipe_identity,
            "material_composition_identity": material_composition_identity,
        }
    )


class MFAEvaluationSession:
    """Calculate ordinary and candidate scenarios from one retained historical prefix."""

    def __init__(
        self,
        request: RemindMFAExecutionRequest,
        parameter_catalogue: tuple[ParameterDefinition, ...],
        *,
        recipe: EcoinventMaterialRecipe,
        composition: MaterialComposition,
    ) -> None:
        """Admit the source, default cohort and scientific input authorities once."""
        self._adapter = RemindMFAExecutionSession(request)
        self._parameter_catalogue = parameter_catalogue
        self._recipe = recipe
        self._composition = composition
        self._cohort_cache: OrderedDict[tuple[str, bool], tuple[ProspectiveCohortResult, str]] = (
            OrderedDict()
        )
        self._cohort_projection_cache: OrderedDict[str, tuple[pd.DataFrame, pd.DataFrame]] = (
            OrderedDict()
        )
        self._authority_identity = mfa_evaluation_session_identity(
            request,
            parameter_catalogue,
            material_recipe_identity=recipe.identity,
            material_composition_identity=composition.identity,
        )
        self._default_cohort = accepted_default_cohort_result(self._adapter.cohort_state)
        self._history: _HistoricalMFAState | None = None
        self._history_stocks: MFAStockState | None = None

    def _prepare_history(self, inputs: RemindMFAProcessInputs) -> _HistoricalMFAState:
        """Prepare the unchanged historical prefix from the first requested scenario.

        Scenario admission preserves all inputs through 2019. Their historical
        MFA calculation supplies the opening state. Retain
        history diagnostics even when a candidate precedes ordinary reporting.
        """
        if self._history is not None:
            return self._history
        history = calculate_mfa_flows(
            self._default_cohort,
            inputs,
            recipe=self._recipe,
            composition=self._composition,
            first_year=1950,
            last_year=2019,
        )
        flow_signature, nominal_signature = _flow_signatures(history)
        # Candidates need 2019 for cumulative impacts, with earlier carbon
        # custody already retained by their admitted foreground parent.
        reporting_boundary = replace(
            history,
            years=(2019,),
            flows={name: values[-1:] for name, values in history.flows.items()},
            material_states=tuple(item for item in history.material_states if item.year == 2019),
            nominal_feedstock_equivalents={
                name: values[-1:] for name, values in history.nominal_feedstock_equivalents.items()
            },
            reconciliation_ledger=pd.DataFrame(),
        )
        self._history = _HistoricalMFAState(
            history,
            reporting_boundary,
            flow_signature,
            nominal_signature,
            _material_signature(history.material_states),
        )
        return self._history

    @property
    def authority_identity(self) -> str:
        """Return the immutable source, input and method authority of this session."""
        return self._authority_identity

    def fixed_lifetime_stock_system(self, *, compiled: pd.DataFrame) -> FixedLifetimeStockSystem:
        """Form proposal constraints from this session's existing cohort history.

        The compiled lifetime schedule supplies the conditional stock constraints
        while the retained historical cohorts remain fixed.
        """
        return fixed_lifetime_stock_system(self._adapter.cohort_state, compiled)

    def _calculate(
        self,
        *,
        compiled: pd.DataFrame,
        record_diagnostics: bool,
        progress: Callable[[str, int | None], None] | None,
    ) -> _ValidatedMFAState:
        """Continue the cohort state and direct MFA calculation."""
        translated = translate_annual_inputs(compiled, self._parameter_catalogue)
        changed = _changed_parameter_ids(compiled)
        cohort_parameters = {
            "stock_projection",
            "lifetime_mean",
            "lifetime_std",
            "plastic_family_share",
            "population",
            "gdppc",
        }
        changed_cohort = changed.intersection(cohort_parameters)
        cohort_identity = canonical_table_sha256(
            compiled.loc[compiled["parameter_id"].isin(cohort_parameters)].reset_index(drop=True)
        )
        cache_key = (cohort_identity, record_diagnostics)
        cached = self._cohort_cache.get(cache_key)
        if cached is None:
            if changed_cohort:
                explicit_stock = (
                    compiled["parameter_id"].eq("stock_projection")
                    & compiled["value_origin"].eq(OPTIMIZATION_PARENT_OVERLAY_ORIGIN)
                ).any()
                target = (
                    self._adapter.resolve_stock_target(translated.numeric_updates)
                    if (changed.intersection({"population", "gdppc"}) and not explicit_stock)
                    else None
                )
                cohort = continue_cohort_stock(
                    self._adapter.cohort_state,
                    compiled,
                    stock_target=target,
                    transition_start_year=_first_changed_year(compiled, set(changed_cohort)),
                    optimization_overlay=bool(
                        compiled["value_origin"].eq(OPTIMIZATION_PARENT_OVERLAY_ORIGIN).any()
                    ),
                    record_diagnostics=record_diagnostics,
                    progress=progress,
                )
            else:
                cohort = self._default_cohort
            cached = (cohort, _cohort_signature(cohort))
            self._cohort_cache[cache_key] = cached
        cohort, stock_signature = cached
        self._cohort_cache.move_to_end(cache_key)
        if len(self._cohort_cache) > _OPTIMIZATION_COHORT_CACHE_SIZE:
            self._cohort_cache.popitem(last=False)
        # With no intervention, preserve the admitted source defaults instead
        # of writing compiled default rows back into native parameters.
        inputs = self._adapter.process_inputs(
            translated.numeric_updates if changed else {},
            carbon_inflow=cohort.carbon_inflow,
            carbon_outflow=cohort.carbon_outflow,
            deferred_waste_trade=translated.deferred_waste_trade if changed else pd.DataFrame(),
            record_diagnostics=record_diagnostics,
        )
        history = self._prepare_history(inputs)
        if progress is not None:
            progress("physical_material_realization", 2020)
        future = calculate_mfa_flows(
            cohort,
            inputs,
            recipe=self._recipe,
            composition=self._composition,
            first_year=2020,
            last_year=2100,
            record_diagnostics=record_diagnostics,
        )
        flow_signature, nominal_signature = _flow_signatures(future)
        # One authority binds the checked MFA state. Ordinary diagnostic
        # Candidate calculation uses its array state and method identity.
        physical_state_identity = (
            MFA_PHYSICAL_STATE_SCHEMA
            + "::"
            + payload_sha256(
                {
                    "schema_version": MFA_PHYSICAL_STATE_SCHEMA,
                    "source_and_method": self.authority_identity,
                    "historical_flows": history.flow_signature,
                    "future_flows": flow_signature,
                    "cohorts": stock_signature,
                    "historical_materials": history.material_signature,
                    "future_materials": _material_signature(future.material_states),
                    "historical_nominal_feedstock": history.nominal_signature,
                    "future_nominal_feedstock": nominal_signature,
                }
            )
        )
        return _ValidatedMFAState(
            cohort,
            cohort_identity,
            future,
            history,
            inputs.waste_trade_propensity_ledger,
            physical_state_identity,
        )

    def evaluate(
        self,
        *,
        foreground_scenario: str,
        compiled: pd.DataFrame,
        progress: Callable[[str, int | None], None] | None = None,
    ) -> MFACandidateEvaluation:
        """Return compact MFA inputs from the same validated direct calculation."""
        state = self._calculate(compiled=compiled, record_diagnostics=False, progress=progress)
        projection = self._cohort_projection_cache.get(state.cohort_identity)
        if projection is None:
            cohort_state = _optimization_cohort_state_table(state.cohort)
            projection = (
                cohort_state,
                _optimization_cohort_material_identity_table(state.cohort, cohort_state),
            )
            self._cohort_projection_cache[state.cohort_identity] = projection
        self._cohort_projection_cache.move_to_end(state.cohort_identity)
        if len(self._cohort_projection_cache) > _OPTIMIZATION_COHORT_CACHE_SIZE:
            self._cohort_projection_cache.popitem(last=False)
        parts = (state.history.reporting_boundary, state.future)
        return MFACandidateEvaluation(
            pd.concat(
                [
                    mfa_flow_table(
                        part,
                        state.cohort,
                        foreground_scenario=foreground_scenario,
                        describe_flows=False,
                    )
                    for part in parts
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    material_realization_table(item)
                    for part in parts
                    for item in part.material_states
                ],
                ignore_index=True,
            ),
            pd.concat([nominal_feedstock_table(part) for part in parts], ignore_index=True),
            projection[0],
            projection[1],
            state.physical_state_identity,
            state.cohort.material_identity_method,
        )

    def compute(
        self,
        *,
        foreground_scenario: str,
        compiled: pd.DataFrame,
        scenario_change_points: Sequence[Mapping[str, object]],
        progress: Callable[[str, int | None], None] | None = None,
    ) -> MFAComputation:
        """Format the complete direct result for ordinary or selected-scenario publication."""
        state = self._calculate(compiled=compiled, record_diagnostics=True, progress=progress)
        history = state.history.flows
        if self._history_stocks is None:
            opening = {
                name: np.zeros((len(history.regions), len(history.materials)))
                for name in ACCUMULATED_STOCKS
            }
            self._history_stocks = accumulate_mfa_stocks(
                self._default_cohort, history, opening=opening
            )
            _require_validity(self._history_stocks.validation)
        future_stocks = accumulate_mfa_stocks(
            state.cohort, state.future, opening=self._history_stocks.closing
        )
        _require_validity(future_stocks.validation)
        parts = (history, state.future)
        model_flows = pd.concat(
            [
                mfa_flow_table(part, state.cohort, foreground_scenario=foreground_scenario)
                for part in parts
            ],
            ignore_index=True,
        )
        model_stocks = pd.concat(
            [
                mfa_stock_table(stocks, part, state.cohort, foreground_scenario=foreground_scenario)
                for stocks, part in ((self._history_stocks, history), (future_stocks, state.future))
            ],
            ignore_index=True,
        )
        stock_checks = pd.concat(
            (
                self._history_stocks.validation.assign(first_year=1950, last_year=2019),
                future_stocks.validation.assign(first_year=2020, last_year=2100),
            ),
            ignore_index=True,
        )
        edges, nodes = build_mfa_sankey(model_flows)
        if progress is not None:
            progress("result_tables", 2100)
        return MFAComputation(
            model_flows,
            model_stocks,
            pd.concat(
                [
                    material_realization_table(item)
                    for part in parts
                    for item in part.material_states
                ],
                ignore_index=True,
            ),
            pd.concat([nominal_feedstock_table(part) for part in parts], ignore_index=True),
            pd.concat([part.reconciliation_ledger for part in parts], ignore_index=True),
            stock_checks,
            state.cohort.transition_ledger,
            state.waste_trade_propensity_ledger,
            _resolved_input_view(
                compiled, scenario_change_points, state.waste_trade_propensity_ledger
            ),
            _derived_outputs(state.cohort, model_flows),
            _cohort_state_table(state.cohort, foreground_scenario=foreground_scenario),
            _cohort_material_identity_table(state.cohort, foreground_scenario=foreground_scenario),
            _cohort_composition_table(state.cohort),
            edges,
            nodes,
            state.physical_state_identity,
            state.cohort.material_identity_method,
        )
