from tempfile import TemporaryFile

import numpy as np
import pandas as pd

from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_lca.calculation.result_diagnostics import (
    _ACTIVATION_LINEAGE_COLUMNS,
    _activation_lineage,
    restore_activation_lineage,
    spool_activation_lineage,
)
from pytoce.run_lca.calculation.results import (
    _restore_resolved_activations,
    _spool_resolved_activations,
)
from pytoce.run_lca.calculation.timex import PrincipalTimeline


def test_resolved_activation_spool_preserves_exact_reporting_parent() -> None:
    activations = pd.DataFrame(
        {
            "year": pd.Series((2025, 2030), dtype="int64"),
            "amount": np.asarray((-0.0, -1.25), dtype=np.float64),
            "mfa_carbon_amount_kg": np.asarray((0.0, 2.5), dtype=np.float64),
            "provider_product_id": pd.Series((17, 23), dtype="int64"),
            "life_cycle_stage": pd.Series(
                ("virgin_resin_production", "chemical_recycling_conversion"),
                dtype="object",
            ),
            "ultimate_carbon_origin": pd.Series(("", "biogenic"), dtype="object"),
            "resolution_status": pd.Series(("", "source_exact"), dtype="object"),
        }
    )

    with TemporaryFile(suffix=".parquet") as spool:
        _spool_resolved_activations(activations, spool)
        restored = _restore_resolved_activations(
            spool,
            columns=tuple(activations.columns),
            life_cycle_stages=(
                "virgin_resin_production",
                "chemical_recycling_conversion",
            ),
        )
        projected = _restore_resolved_activations(
            spool,
            columns=("year", "life_cycle_stage", "amount"),
            life_cycle_stages=("chemical_recycling_conversion",),
        )

    pd.testing.assert_frame_equal(restored, activations, check_exact=True)
    pd.testing.assert_frame_equal(
        projected,
        activations.loc[
            activations["life_cycle_stage"].eq("chemical_recycling_conversion"),
            ["year", "life_cycle_stage", "amount"],
        ].reset_index(drop=True),
        check_exact=True,
    )
    assert np.signbit(restored.loc[0, "amount"])


def test_activation_lineage_spool_preserves_exact_sorted_table() -> None:
    derived = {
        "foreground_scenario",
        "background_scenario",
        "ssp_scenario",
        "h12_region",
        "provider_member",
        "final_provider_demand_amount_kg",
        "provider_demand_unit",
    }
    integer_columns = {
        "year",
        "physical_event_year",
        "provider_anchor_year",
        "general_background_anchor_year",
    }
    float_columns = {
        "source_parent_amount_kg",
        "source_allocation_share",
        "origin_fraction",
        "material_component_share",
        "infiltration_share",
        "throughput_multiplier",
        "styrene_component_share",
        "butadiene_analogue_component_share",
        "sbr_share_in_material",
        "temporal_provider_share",
        "mfa_carbon_amount_kg",
    }
    boolean_columns = {
        "whole_tyre_constituents_excluded",
        "no_treatment_sensitivity_executed",
    }
    source_columns = [column for column in _ACTIVATION_LINEAGE_COLUMNS if column not in derived]
    source_columns.extend(("region", "amount"))
    timelines: dict[str, PrincipalTimeline] = {}
    for index, region in enumerate(H12_REGIONS):
        year = 2030 if index % 2 == 0 else 2025
        row: dict[str, object] = {}
        for column in source_columns:
            if column == "region":
                row[column] = region
            elif column == "amount":
                row[column] = -0.0 if index == 0 else float(index + 1)
            elif column == "year" or column == "physical_event_year":
                row[column] = year
            elif column == "provider_anchor_year":
                row[column] = 2030
            elif column == "general_background_anchor_year":
                row[column] = 2025
            elif column in float_columns:
                row[column] = float(index + 1) / 12.0
            elif column in boolean_columns:
                row[column] = False
            elif column in integer_columns:
                row[column] = year
            else:
                row[column] = f"{column}_{index}"
        timelines[region] = PrincipalTimeline(
            timeline=pd.DataFrame(),
            activations=pd.DataFrame([row]),
            base_static_score=0.0,
            direct_inventory=pd.DataFrame(columns=("year", "flow_id", "amount")),
        )

    expected = _activation_lineage(
        pd.concat(
            [timelines[region].activations for region in H12_REGIONS],
            ignore_index=True,
        ),
        foreground_scenario="remind_mfa_default",
        background_scenario="remind_ssp2_npi",
    )
    with TemporaryFile(suffix=".parquet") as spool:
        spool_activation_lineage(
            timelines,
            spool,
            foreground_scenario="remind_mfa_default",
            background_scenario="remind_ssp2_npi",
        )
        actual = restore_activation_lineage(spool)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
