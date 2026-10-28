"""Defaults for optional metadata in prepared material-to-provider mappings."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from material_carbon_fixtures import prepared_material_carbon

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
)
from pytoce.run_lca.calculation.providers import load_provider_routes
from tests.integration.brightway_lca_fixtures import write_prepared_inputs


def test_stored_optional_route_metadata_uses_declared_defaults(tmp_path: Path) -> None:
    """Read nullable route metadata while preserving material, carbon origin and product."""
    authority = prepared_provider_authority()
    rows = authority.routes.loc[
        authority.routes.provider_demand_type.eq("processing_conversion")
        & authority.routes.material.eq("PP")
    ].to_dict("records")
    defaults = {
        "lca_fate": "",
        "infiltration_class": "",
        "provider_class": "",
        "analogue_status": "",
        "resolution_status": "",
        "material_functional_unit": "",
        "sbr_proxy_component_role": "",
        "source_role": "",
        "material_component_share": 1.0,
        "infiltration_share": 1.0,
        "styrene_component_share": 0.0,
        "butadiene_analogue_component_share": 0.0,
        "sbr_share_in_material": 0.0,
        "whole_tyre_constituents_excluded": False,
        "no_treatment_sensitivity_executed": False,
    }
    for row in rows:
        for column in defaults:
            row[column] = None
    provider = replace(authority, routes=pd.DataFrame.from_records(rows))
    shared_manifest, _ = write_prepared_inputs(
        tmp_path,
        BrightwayNames(
            tmp_path / "data_processed/brightway", "route_defaults", "biosphere", "native"
        ),
        methods=pd.DataFrame(),
        provider=provider,
        carbon_rows=prepared_material_carbon(tmp_path / "native_carbon"),
        backgrounds=(),
    )
    shared = shared_manifest.path.parent
    original_bytes = (shared / "provider_service_routes.parquet").read_bytes()
    loaded = load_provider_routes(shared_manifest)
    resolved = loaded.resolve(
        provider_demand_type="processing_conversion",
        material="PP",
        constituent="pp",
        product="polymer",
        year=2050,
        mapping_basis="kg_processed_plastic",
        virgin_feedstock_route="all",
        ultimate_carbon_origin="fossil",
    )
    for component in resolved:
        assert {name: getattr(component, name) for name in defaults} == defaults
        assert component.virgin_feedstock_route == "all"
        assert component.ultimate_carbon_origin == "fossil"
        assert component.constituent == "pp"
        assert component.product == "polymer"
    assert (shared / "provider_service_routes.parquet").read_bytes() == original_bytes
