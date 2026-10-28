"""Build the reusable optimization response bank from prepared backgrounds."""

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    read_prepared_manifest,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
)
from pytoce.run_lca.calculation.lcia import (
    LCIAMethodSelection,
)
from pytoce.run_lca.calculation.providers import (
    PreparedProviderProduct,
    load_prepared_provider_product,
    prepared_brightway_names,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _BACKGROUNDS,
    _METHOD_SIGNATURE_COLUMNS,
    OptimizationResponseBank,
    ResponseBankInputs,
    ResponseBankMemberParent,
)
from pytoce.run_optimization.calculation.response_bank_store import (
    _owner_lock,
    _resolved_owner,
    load_optimization_response_bank,
    write_optimization_response_bank,
)
from pytoce.shared.runtime.filesystem import (
    canonical_table_sha256,
    payload_sha256,
)

from .response_kernel import (
    _BUILDER_METHOD_IDENTITY,
    _RESPONSE_BANK_READABLE_VERSION,
    ResponseMethodSelection,
    _direct_factor_table,
    _member_response,
    _resolved_response_methods,
    _service_registry,
)


def _response_slice_readable_name(
    method_tuple: tuple[str, ...],
) -> str:
    """Return a bounded readable name for an admitted exact PB-LCIA method."""

    category = method_tuple[2]
    characters = "".join(
        character if character.isascii() and character.isalnum() else " " for character in category
    )
    category_slug = "_".join(characters.lower().split())[:40].rstrip("_")
    discriminator = json.dumps(method_tuple, ensure_ascii=True, separators=(",", ":"))
    return f"pb_lcia_method_{sha256(discriminator.encode()).hexdigest()[:10]}_{category_slug}"


def response_method_selection(
    result_lcia_method: LCIAMethodSelection,
) -> ResponseMethodSelection:
    """Combine required EF constraints with the admitted selected-result methods.

    All sixteen EF categories are required for optimization. Reporting may add
    the complete PB group or one exact PB method. An exact EF reporting method
    is already covered. A completed full bank can satisfy any of these requests.
    Public request normalization admits exact methods against the package catalogue;
    prepared authorities supply their factors.
    """
    groups = ("ef31", "pb_lcia") if "pb_lcia" in result_lcia_method.groups else ("ef31",)
    selections = (LCIAMethodSelection(groups=groups, exact_method=()),)
    name = "_".join(groups)
    if result_lcia_method.exact_method and result_lcia_method.exact_method[0] == "AESA (PBs-LCIA)":
        selections = (*selections, result_lcia_method)
        name += "_and_" + _response_slice_readable_name(result_lcia_method.exact_method)
    return ResponseMethodSelection(
        readable_name=f"{name}_{_RESPONSE_BANK_READABLE_VERSION}",
        selections=selections,
        groups=groups,
    )


def ensure_optimization_response_bank(
    workspace: Path,
    *,
    selection: ResponseMethodSelection,
    member_progress: Callable[[str], None],
) -> OptimizationResponseBank:
    """Reuse or build one exact method-union provider-response product."""

    existing = _available_response_bank(workspace, selection.readable_name)
    if existing is not None:
        return existing
    owner = _resolved_owner(workspace)
    with _owner_lock(owner):
        return _ensure_optimization_response_bank(
            workspace,
            owner=owner,
            selection=selection,
            member_progress=member_progress,
        )


def _available_response_bank(
    workspace: Path,
    name: str,
) -> OptimizationResponseBank | None:
    """Return the requested completed bank or the complete standard superset."""

    existing = load_optimization_response_bank(workspace, name)
    if existing is not None:
        return existing
    full_union_name = f"ef31_pb_lcia_{_RESPONSE_BANK_READABLE_VERSION}"
    if name != full_union_name:
        accepted_superset = load_optimization_response_bank(workspace, full_union_name)
        if accepted_superset is not None:
            return accepted_superset
    return None


def _ensure_optimization_response_bank(
    workspace: Path,
    *,
    owner: Path,
    selection: ResponseMethodSelection,
    member_progress: Callable[[str], None],
) -> OptimizationResponseBank:
    """Build or reuse the provider response tables used by optimization.

    For each of the four prepared REMIND background scenarios, the function
    calculates the LCIA response of one unit of every foreground provider
    service and the Timex weights that map anchor-year responses to activity years. A complete
    response result covering the
    requested LCIA methods is reused.

    Args:
        workspace: Active pyToCE workspace containing prepared backgrounds.
        owner: Prepared response-bank directory held under its writer lock.
        selection: Exact LCIA methods required for candidate and selected results.
        member_progress: Callback receiving each background scenario name.

    Returns:
        The response bank's status, content identifier, LCIA method selection name,
        and directory containing the saved tables.

    Raises:
        PreparedDataError: If a required prepared manifest or Brightway store is missing.
        ResponseBuilderError: If a prepared provider service is unavailable.
    """

    name = selection.readable_name
    existing = _available_response_bank(workspace, name)
    if existing is not None:
        return existing

    shared = workspace / "data_processed" / "prepared_backgrounds" / "shared"
    shared_reference = read_prepared_manifest(shared / "manifest.json")
    shared_manifest = shared_reference.payload
    shared_sha256 = shared_reference.sha256
    service_manifest = json.loads(
        (shared_reference.path.parent / "provider_service_manifest.json").read_bytes()
    )
    routes = pq.read_table(
        shared_reference.path.parent / "provider_service_routes.parquet"
    ).to_pandas()
    services = _service_registry(routes)
    common_impact_categories = pq.read_table(
        shared_reference.path.parent / "impact_categories.parquet"
    ).to_pandas()
    member_parents: list[ResponseBankMemberParent] = []
    member_calculations: list[tuple[BrightwayNames, PreparedProviderProduct]] = []
    for background in _BACKGROUNDS:
        member = workspace / "data_processed" / "prepared_backgrounds" / background
        member_reference = read_prepared_manifest(member / "manifest.json")
        manifest = member_reference.payload
        manifest_sha256 = member_reference.sha256
        names = prepared_brightway_names(workspace, shared_reference)
        providers = load_prepared_provider_product(shared_reference, member_reference)
        member_calculations.append((names, providers))
        member_parents.append(
            ResponseBankMemberParent(
                background_scenario=background,
                prepared_background_identity=str(manifest["identity"]),
                prepared_background_manifest_sha256=manifest_sha256,
            )
        )
    # Every member has the same admitted method authority and Brightway project.
    # Resolve that external method registration once before numerical responses.
    methods, method_registry = _resolved_response_methods(
        member_calculations[0][0], selection, common_impact_categories
    )
    response_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    native_frames: list[pd.DataFrame] = []
    direct_flows: dict[tuple[str, str], PaReceivingFlow] = {}
    for background, (names, providers) in zip(_BACKGROUNDS, member_calculations, strict=True):
        member_progress(background)
        response = _member_response(
            names, background, methods, method_registry, services, providers
        )
        response_frames.append(response.provider_unit_responses)
        weight_frames.append(response.timex_year_weights)
        native_frames.append(response.native_unit_responses)
        direct_flows.update((flow.key, flow) for flow in response.direct_flows)
    responses = pd.concat(response_frames, ignore_index=True).sort_values(
        [
            "background_scenario",
            "provider_anchor_year",
            "provider_service_id",
            *_METHOD_SIGNATURE_COLUMNS,
        ],
        kind="stable",
        ignore_index=True,
    )
    weights = pd.concat(weight_frames, ignore_index=True).sort_values(
        ["background_scenario", "occurrence_year", "provider_anchor_year"],
        kind="stable",
        ignore_index=True,
    )
    native = pd.concat(native_frames, ignore_index=True)
    direct = _direct_factor_table(
        member_calculations[0][0],
        methods,
        method_registry,
        tuple(flow for _, flow in sorted(direct_flows.items())),
    )
    climate = method_registry.loc[
        method_registry["impact_category_tuple_json"].eq(
            '["EF v3.1","climate change","global warming potential (GWP100)"]'
        )
    ].reset_index(drop=True)
    source_payload = {
        "method_identity": _BUILDER_METHOD_IDENTITY,
        "response_method_union": name,
        "provider_route_identity": service_manifest["identity"],
        "shared_prepared_identity": shared_manifest["identity"],
        "member_parents": [
            {
                "background_scenario": parent.background_scenario,
                "identity": parent.prepared_background_identity,
                "manifest_sha256": parent.prepared_background_manifest_sha256,
            }
            for parent in member_parents
        ],
        "method_registry_identity": canonical_table_sha256(method_registry),
        "provider_unit_responses_identity": canonical_table_sha256(responses),
        "timex_year_weights_identity": canonical_table_sha256(weights),
        "native_unit_responses_identity": canonical_table_sha256(native),
        "direct_characterization_factors_identity": canonical_table_sha256(direct),
    }
    source_identity = _BUILDER_METHOD_IDENTITY + "::" + payload_sha256(source_payload)
    source = ResponseBankInputs(
        provider_route_identity=str(service_manifest["identity"]),
        prepared_background_shared_identity=str(shared_manifest["identity"]),
        prepared_background_shared_manifest_sha256=shared_sha256,
        builder_identity=source_identity,
        member_parents=tuple(member_parents),
    )
    return write_optimization_response_bank(
        owner,
        response_method_union=name,
        method_union_registry=method_registry,
        mandatory_climate_method_registry=climate,
        provider_service_registry=services.assign(
            provider_unit=responses.groupby("provider_service_id", sort=False)["provider_unit"]
            .first()
            .reindex(services["provider_service_id"])
            .to_numpy()
        ).astype("string"),
        provider_unit_responses=responses,
        timex_year_weights=weights,
        native_unit_responses=native,
        direct_characterization_factors=direct,
        source_inputs=source,
    )
