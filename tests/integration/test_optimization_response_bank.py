"""Optimization response construction through real prepared files and Brightway."""

import json
from dataclasses import replace
from hashlib import sha256
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from pytoce import set_workspace
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.premise import (
    ACTIVE_BACKGROUND_SCENARIOS,
)
from pytoce.prepare_backgrounds.preparation.project_biosphere import load_lcia_catalogue
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    provider_frontier_database_name,
)
from pytoce.run_lca.calculation.lcia import (
    normalized_lcia_method,
    resolve_lcia_methods_from_authority,
)
from pytoce.run_optimization.calculation.response_bank import (
    read_optimization_response_bank,
)
from pytoce.run_optimization.calculation.response_builder import (
    ensure_optimization_response_bank,
    response_method_selection,
)
from pytoce.run_optimization.calculation.response_kernel import (
    _resolved_response_methods,
)
from tests.integration.brightway_lca_fixtures import (
    _PROVIDER_SERVICE_CODE,
    _PROVIDER_SERVICE_ID,
    PROVIDER_TECHNICAL_YEARS,
    _process,
    write_prepared_inputs,
)
from tests.integration.pa_background_fixtures import (
    PaBackgroundInputs,
    prepare_pa_background_inputs,
)

_BACKGROUNDS = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)


def _method_records(lcia_groups: tuple[str, ...]) -> list[dict[str, object]]:
    """Return the selected package methods with deterministic test factors."""

    specifications, _ = load_lcia_catalogue()
    return [
        {
            "selection_order": method.order,
            "method_id": method.method_id,
            "method_version": method.method_version,
            "impact_category_id": method.impact_category_id,
            "impact_category_tuple_json": json.dumps(
                method.impact_category_tuple, separators=(",", ":")
            ),
            "unit": method.unit,
            "factor_identity": sha256(
                json.dumps((method.impact_category_tuple, method.order - 2.0)).encode("utf-8")
            ).hexdigest(),
        }
        for method in specifications
        if method.method_id in lcia_groups
    ]


def _prepared_artifacts(
    workspace: Path, receiving: PaBackgroundInputs, lcia_groups: tuple[str, ...]
) -> None:
    """Publish all four members through actual shared/member table writers."""
    provider = PreparedProviderAuthority(
        identity="provider-authority::response-fixture",
        interface_database="provider_interface_response_fixture",
        catalogue={"schema_version": "bounded_provider_catalogue_schema_v1"},
        routes=pd.DataFrame(
            [
                {
                    "provider_service_id": _PROVIDER_SERVICE_ID,
                    "provider_service_code": _PROVIDER_SERVICE_CODE,
                }
            ]
        ),
        source_authorities=(),
        method_authorities=(),
    )
    write_prepared_inputs(
        workspace,
        receiving.names,
        methods=pd.DataFrame(_method_records(lcia_groups)),
        provider=provider,
        carbon_rows=(),
        backgrounds=_BACKGROUNDS,
        receiving_reference=receiving.reference,
        receiving_sources_by_background={
            background: {
                year: replace(
                    receiving.original if year <= 2025 else receiving.future,
                    background_scenario=background,
                    provider_anchor_year=year,
                    general_background_anchor_year=2025 if year <= 2025 else year,
                )
                for year in PROVIDER_TECHNICAL_YEARS
            }
            for background in _BACKGROUNDS
        },
        technology_databases_by_background={
            background: {year: "future" for year in PROVIDER_TECHNICAL_YEARS}
            for background in _BACKGROUNDS
        },
    )


def _brightway_fixture(names: BrightwayNames, lcia_groups: tuple[str, ...]) -> None:
    """Add one unit service per anchor to the existing complete PA source graph."""

    biosphere = (names.biosphere_database, "co2")
    with selected_brightway_project(names) as bd:
        bd.preferences["biosphere_database"] = names.biosphere_database
        for background, year in product(_BACKGROUNDS, PROVIDER_TECHNICAL_YEARS):
            database = provider_frontier_database_name(
                background, year, interface_database="provider_interface_response_fixture"
            )
            bd.Database(database).write(
                {
                    (database, _PROVIDER_SERVICE_CODE): _process(
                        database,
                        _PROVIDER_SERVICE_CODE,
                        biosphere=biosphere,
                    )
                },
                searchable=False,
            )
        for method in _method_records(lcia_groups):
            method_tuple = tuple(json.loads(str(method["impact_category_tuple_json"])))
            bd.Method(method_tuple).register(unit=str(method["unit"]))
            bd.Method(method_tuple).write([(biosphere, float(method["selection_order"]) - 2.0)])


@pytest.mark.scientific_integration
@pytest.mark.parametrize("report_method", ["ef31", "ef31_pb_lcia", "exact_pb"])
def test_response_bank_builds_four_real_brightway_members(
    tmp_path: Path, report_method: str
) -> None:
    """Build provider responses for all four backgrounds."""

    workspace = set_workspace(tmp_path / "workspace").workspace
    receiving = prepare_pa_background_inputs(workspace)
    names = receiving.names
    lcia_groups = ("ef31",) if report_method == "ef31" else ("ef31", "pb_lcia")
    _prepared_artifacts(workspace, receiving, lcia_groups)
    _brightway_fixture(names, lcia_groups)
    lcia_selection = normalized_lcia_method(lcia_groups)
    baseline_selection = response_method_selection(lcia_selection)
    if report_method == "exact_pb":
        specifications, _ = load_lcia_catalogue()
        exact_pb = next(method for method in specifications if method.method_id == "pb_lcia")
        selection = response_method_selection(
            normalized_lcia_method(exact_pb.impact_category_tuple)
        )
    else:
        selection = baseline_selection
    shared = workspace / "data_processed/prepared_backgrounds"
    resolved_methods, method_registry = _resolved_response_methods(
        names,
        baseline_selection,
        pd.read_parquet(shared / _BACKGROUNDS[0] / "impact_categories.parquet"),
    )
    # Ordinary exact selections and response projections must share the same
    # prepared method labels, unit and factor authority for every category.
    with selected_brightway_project(names) as bd:
        for method in resolved_methods:
            ordinary = resolve_lcia_methods_from_authority(
                pd.read_parquet(shared / _BACKGROUNDS[0] / "impact_categories.parquet"),
                normalized_lcia_method(method.method_tuple),
                bd,
            )
            assert ordinary == (replace(method, selection_order=1),)
    progress: list[str] = []

    result = ensure_optimization_response_bank(
        workspace,
        selection=selection,
        member_progress=progress.append,
    )

    assert result.status == "computed"
    assert progress == list(_BACKGROUNDS)
    assert result.result_directory.is_dir()
    loaded = read_optimization_response_bank(result)
    specifications, _ = load_lcia_catalogue()
    expected_specs = [
        method
        for method in specifications
        if method.method_id == "ef31"
        or report_method == "ef31_pb_lcia"
        or report_method == "exact_pb"
        and method == exact_pb
    ]
    expected_specs.sort(
        key=lambda method: (
            method.method_id
            if method.method_id in selection.groups
            else json.dumps(method.impact_category_tuple, separators=(",", ":")),
            method.order,
        )
    )
    assert loaded.method_registry["impact_category_tuple_json"].tolist() == [
        json.dumps(method.impact_category_tuple, separators=(",", ":")) for method in expected_specs
    ]
    assert loaded.method_registry["unit"].tolist() == [method.unit for method in expected_specs]
    expected_factors = {
        json.dumps(method.impact_category_tuple, separators=(",", ":")): method.order - 2.0
        for method in expected_specs
    }
    observed = loaded.provider_unit_responses
    expected_scores = observed["impact_category_tuple_json"].map(expected_factors)
    pd.testing.assert_series_equal(observed["unit_score"], expected_scores, check_names=False)
    pd.testing.assert_series_equal(
        observed["unit_contribution_norm"], expected_scores.abs(), check_names=False
    )

    assert set(observed["background_scenario"]) == set(_BACKGROUNDS)
    assert set(observed["provider_anchor_year"]) == set(PROVIDER_TECHNICAL_YEARS)
    assert len(observed) == len(_BACKGROUNDS) * len(PROVIDER_TECHNICAL_YEARS) * len(expected_specs)
    native = loaded.native_unit_responses
    assert set(native["background_scenario"]) == set(_BACKGROUNDS)
    assert set(native["source_anchor_year"]) == set(PROVIDER_TECHNICAL_YEARS) - {2020}
    # The reused native fixture supplies complete PAM with 0.1 kg CO2/kg.
    # This checks a real PA endpoint solve, including all methods and dates.
    pam = native.loc[native["endpoint_code"].eq("mc_pam")]
    assert set(pam["endpoint_database"]) == {"original", "future"}
    assert len(pam) == len(_BACKGROUNDS) * (len(PROVIDER_TECHNICAL_YEARS) - 1) * len(expected_specs)
    assert pam["unit_score"].to_numpy() == pytest.approx(
        0.1 * pam["impact_category_tuple_json"].map(expected_factors).to_numpy()
    )
    assert set(loaded.timex_year_weights["occurrence_year"]) == set(range(2019, 2101))
    assert (
        loaded.timex_year_weights.groupby(["background_scenario", "occurrence_year"])[
            "temporal_provider_share"
        ]
        .sum()
        .eq(1.0)
        .all()
    )
