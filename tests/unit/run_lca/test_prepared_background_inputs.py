"""Prepared background scenarios, dated suppliers and scientific result labels."""

from pathlib import Path

import pandas as pd
import pytest
from infiltration_fixtures import resolved_infiltration_provider

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    PreparedDataError,
)
from pytoce.run_lca.calculation.providers import (
    load_prepared_provider_product,
    prepared_brightway_names,
    prepared_store_directory,
)
from pytoce.run_lca.run_lca import _background_dimensions
from tests.integration.brightway_lca_fixtures import PROVIDER_TECHNICAL_YEARS, write_prepared_inputs


@pytest.fixture
def prepared_inputs(tmp_path: Path):
    """Write a synthetic prepared background with a recorded characterization method."""
    names = BrightwayNames(
        tmp_path / "data_processed/brightway", "fixture_project", "biosphere", "native"
    )
    names.store.mkdir(parents=True)
    shared, members = write_prepared_inputs(
        tmp_path,
        names,
        methods=pd.DataFrame(
            [{"method_id": "fixture", "unit": "impact", "factor_identity": "fixture"}]
        ),
        provider=resolved_infiltration_provider(tmp_path / "native_infiltration"),
        carbon_rows=(),
        backgrounds=("remind_ssp2_npi",),
    )
    return shared, members["remind_ssp2_npi"], names


def test_prepared_inputs_preserve_project_member_technology_and_result_labels(
    tmp_path: Path, prepared_inputs
) -> None:
    """Keep the selected scenario's project, provider years and scientific labels."""
    shared, member, expected_names = prepared_inputs
    names = prepared_brightway_names(tmp_path, shared)
    product = load_prepared_provider_product(shared, member)
    assert names == expected_names
    assert tuple(product.database_by_year) == PROVIDER_TECHNICAL_YEARS
    assert tuple(product.timeline_database_by_year) == PROVIDER_TECHNICAL_YEARS
    assert all("remind_ssp2_npi" in name for name in product.database_by_year.values())
    assert _background_dimensions(member) == {
        name: member.payload[name]
        for name in (
            "ssp_scenario",
            "iam_model",
            "iam_scenario",
            "gmst",
            "gmst_source",
            "premise_version",
            "ecoinvent_version",
            "system_model",
        )
    }


def test_prepared_store_rejects_missing_and_regular_file_locations(tmp_path: Path) -> None:
    """Reject missing paths and files and preserve existing contents."""
    path = tmp_path / "store"
    with pytest.raises(PreparedDataError, match="is unavailable"):
        prepared_store_directory(path)
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(PreparedDataError, match="not a directory"):
        prepared_store_directory(path)
    assert path.read_text(encoding="utf-8") == "preserve"
