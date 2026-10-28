import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources

from pytoce.run_lca.calculation.lcia import (
    normalized_lcia_method,
    resolve_lcia_methods_from_authority,
)
from pytoce.run_lca.calculation.providers import (
    PreparedProviderProduct,
    prepared_store_directory,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    method_registry_from_records,
)
from pytoce.run_optimization.calculation.response_builder import (
    response_method_selection,
)
from pytoce.run_optimization.calculation.response_kernel import (
    _method_registry,
    _timex_weights,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256


def test_response_method_union_is_the_smallest_accepted_union() -> None:
    ef_only = response_method_selection(normalized_lcia_method(("ef31",)))
    ef_and_pb = response_method_selection(normalized_lcia_method(("ef31", "pb_lcia")))

    assert ef_only.readable_name == "ef31_response_v3"
    assert ef_only.groups == ("ef31",)
    assert len(ef_only.selections) == 1
    assert ef_and_pb.readable_name == "ef31_pb_lcia_response_v3"
    assert ef_and_pb.groups == ("ef31", "pb_lcia")
    assert len(ef_and_pb.selections) == 1
    assert response_method_selection(normalized_lcia_method("pb_lcia")) == ef_and_pb


def test_exact_method_inside_an_included_group_does_not_duplicate_the_union() -> None:
    exact = (
        "EF v3.1",
        "climate change",
        "global warming potential (GWP100)",
    )

    selection = response_method_selection(normalized_lcia_method(exact))

    assert selection.groups == ("ef31",)
    assert len(selection.selections) == 1
    assert selection.selections[0].groups == ("ef31",)


def test_exact_method_outside_the_included_group_joins_the_response_union() -> None:
    exact = (
        "AESA (PBs-LCIA)",
        "0.9.2",
        "climate change",
        "atmospheric CO2 concentration",
    )

    selection = response_method_selection(normalized_lcia_method(exact))

    assert selection.groups == ("ef31",)
    assert len(selection.selections) == 2
    assert selection.selections[1].exact_method == exact


def test_provider_technology_timeline_is_projected_without_a_timex_rerun() -> None:
    frontiers = pd.DataFrame(
        {
            "anchor_year": (
                2020,
                2025,
                2030,
                2035,
                2040,
                2045,
                2050,
                2055,
                2060,
                2070,
                2080,
                2090,
                2100,
            )
        }
    )
    weights = _timex_weights(
        PreparedProviderProduct(
            {year: f"provider_{year}" for year in frontiers.anchor_year},
            {},
            pa_receiving_reference=pa_receiving_reference(),
            pa_receiving_sources_by_year={
                int(year): pa_receiving_sources(int(year), background_scenario="remind_ssp2_ndc")
                for year in frontiers.anchor_year
            },
        ),
        "remind_ssp2_ndc",
    )

    assert set(weights["occurrence_year"]) == set(range(2019, 2101))
    assert weights.groupby("occurrence_year")["temporal_provider_share"].sum().eq(1.0).all()
    assert weights.loc[weights["occurrence_year"].eq(2019), "provider_anchor_year"].item() == 2020
    assert weights.loc[weights["occurrence_year"].eq(2021), "temporal_provider_share"].tolist() == [
        0.8,
        0.2,
    ]


def test_exact_prepared_method_factor_comes_from_the_prepared_authority() -> None:
    method_tuple = ("EF v3.1", "climate change", "global warming potential (GWP100)")
    prepared_factor = "a" * 64
    authority = pd.DataFrame.from_records(
        [
            {
                "selection_order": 2,
                "method_id": "ef31",
                "method_version": "3.1",
                "impact_category_id": "climate_change",
                "impact_category_tuple_json": json.dumps(method_tuple, separators=(",", ":")),
                "unit": "kg CO2-Eq",
                "factor_identity": prepared_factor,
            }
        ]
    )
    brightway = SimpleNamespace(
        methods={method_tuple: {"unit": "kg CO2-Eq"}},
        Method=lambda _method: pytest.fail("prepared exact methods must not be re-hashed"),
    )
    resolved = resolve_lcia_methods_from_authority(
        authority,
        normalized_lcia_method(method_tuple),
        brightway,
    )
    registry = _method_registry(resolved, ())
    grouped_registry = _method_registry(resolved, ("ef31",))
    reconstructed = method_registry_from_records(grouped_registry.to_dict("records"))

    assert registry.loc[0, "factor_identity"] == prepared_factor
    assert grouped_registry["impact_category_tuple_json"].dtype == "string"
    pd.testing.assert_frame_equal(grouped_registry, reconstructed, check_exact=True)
    assert canonical_table_sha256(grouped_registry) == canonical_table_sha256(reconstructed)


def test_prepared_brightway_store_allows_workspace_directory_link(tmp_path: Path) -> None:
    target = tmp_path / "shared_store"
    target.mkdir()
    link = tmp_path / "data_processed" / "brightway"
    link.parent.mkdir()
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"This platform cannot create a directory link: {exc}")
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"This platform cannot create a directory link: {exc}")

    assert prepared_store_directory(link) == link
