"""Global scores and shares calculated from the 12 world regions."""

import math

import pandas as pd
import pytest

from pytoce.run_lca.calculation.contribution_accumulation import (
    GLOBAL_REGION,
)
from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_lca.calculation.stage_scores import (
    _with_shares,
    attach_contribution_norms,
    derive_global_contributions,
)


def test_global_norm_sums_regional_absolute_contributions() -> None:
    rows = []
    for index, region in enumerate(H12_REGIONS):
        score = 1.0 if index == 0 else -1.0 if index == 1 else 0.0
        rows.append(
            {
                "region": region,
                "year": 2050,
                "method_id": "pb_lcia",
                "method_version": "0.9.2",
                "impact_category_id": "climate change",
                "impact_category_tuple_json": '["AESA (PBs-LCIA)","climate change"]',
                "unit": "W m-2",
                "contribution_view": "life_cycle_stage",
                "record_role": "detail",
                "life_cycle_stage": "production",
                "score": score,
                "contribution_norm": abs(score),
            }
        )

    contributions = _with_shares(derive_global_contributions(pd.DataFrame(rows)))
    global_row = contributions.loc[contributions["region"].eq(GLOBAL_REGION)].iloc[0]

    assert global_row["score"] == pytest.approx(0.0)
    assert global_row["contribution_norm"] == pytest.approx(2.0)
    assert global_row["share_status"] == "zero_score"
    assert math.isnan(float(global_row["share"]))
    assert global_row["norm_share"] == pytest.approx(1.0)


def _principal_and_contribution_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return matching principal scores and finest-grain annual contributions."""

    method = {
        "method_id": "ef31",
        "method_version": "3.1",
        "impact_category_id": "climate_change",
        "impact_category_tuple_json": '["EF v3.1","climate change"]',
        "unit": "kg CO2-Eq",
    }
    scores = pd.DataFrame(
        [
            {"region": region, **method, "score": float(index - 1)}
            for index, region in enumerate(H12_REGIONS)
        ]
    )
    annual = scores.assign(year=2050)
    norms = annual.assign(
        contribution_norm=annual["score"].abs(),
    )
    return scores, annual, norms


def test_principal_scores_receive_regional_norms_before_global_aggregation() -> None:
    """Attach contribution data and calculate global values once."""

    scores, annual, norms = _principal_and_contribution_tables()

    complete_scores, complete_annual = attach_contribution_norms(scores, annual, norms)

    global_score = complete_scores.loc[complete_scores["region"].eq(GLOBAL_REGION)].iloc[0]
    global_annual = complete_annual.loc[complete_annual["region"].eq(GLOBAL_REGION)].iloc[0]
    assert global_score["score"] == pytest.approx(sum(range(-1, len(H12_REGIONS) - 1)))
    assert global_score["contribution_norm"] == pytest.approx(
        sum(abs(index - 1) for index in range(len(H12_REGIONS)))
    )
    assert global_annual["score"] == pytest.approx(global_score["score"])
    assert global_annual["contribution_norm"] == pytest.approx(global_score["contribution_norm"])
