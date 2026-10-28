"""Summed impact contributions and the explicit unattributed residual."""

from types import ModuleType

import pandas as pd
import pytest

from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    create_provider_response_state,
)
from pytoce.run_lca.calculation.contribution_records import (
    _row,
)
from pytoce.run_lca.calculation.contribution_structure import _structural_attribution
from pytoce.run_lca.calculation.stage_scores import (
    _with_shares,
)


def test_structural_attribution_closes_with_explicit_residual(
    contribution_brightway: ModuleType,
) -> None:
    """Preserve signed stage impacts, absolute contribution totals and a zero residual."""
    state = create_provider_response_state(
        contribution_brightway,
        (
            ContributionMethod(
                method_id="ef31",
                method_version="3.1",
                impact_category_id="climate change",
                method_tuple=("EF v3.1", "climate change"),
                unit="kg CO2-Eq",
            ),
        ),
    )
    contributions = _with_shares(
        pd.DataFrame(
            [
                _row(
                    state,
                    "EUR",
                    2050,
                    0,
                    "life_cycle_stage",
                    "detail",
                    score,
                    abs(score),
                    life_cycle_stage=stage,
                )
                for stage, score in (("production", 2.0), ("recycling", -1.0))
            ]
        )
    )

    attribution = _structural_attribution(contributions)
    residual = attribution.loc[attribution["attribution_role"].eq("explicit_residual")].iloc[0]

    assert attribution["score"].sum() == pytest.approx(1.0)
    assert attribution["contribution_norm"].sum() == pytest.approx(3.0)
    assert residual["score"] == pytest.approx(0.0)
    assert residual["contribution_norm"] == pytest.approx(0.0)
    assert residual["share_status"] == "available"
