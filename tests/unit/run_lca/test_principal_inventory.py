from math import fsum
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("bw2calc")

from pytoce.run_lca.calculation.lcia import (
    ImpactCategory,
    characterize_principal_inventory,
)
from pytoce.run_lca.calculation.timex import (
    CharacterizationFactorBank,
    PrincipalTimeline,
    ProviderInventoryBank,
    ProviderUnitInventory,
    assemble_principal_inventory,
    compact_principal_timelines,
)


def _reference_inventory(
    timeline: PrincipalTimeline,
    bank: ProviderInventoryBank,
) -> pd.DataFrame:
    amounts: dict[tuple[int, int], list[float]] = {}
    for raw_activation in timeline.activations.itertuples(index=False):
        activation = cast(Any, raw_activation)
        unit = bank.inventories[
            (
                str(activation.matrix_database),
                str(activation.provider_database),
                str(activation.provider_code),
            )
        ]
        for flow_id, amount in zip(unit.flow_ids, unit.amounts):
            amounts.setdefault((int(activation.year), int(flow_id)), []).append(
                float(activation.amount) * float(amount)
            )
    rows = []
    for (year, flow_id), values in sorted(amounts.items()):
        amount = fsum(values)
        if amount != 0.0:
            rows.append({"year": year, "flow_id": flow_id, "amount": amount})
    return pd.DataFrame(rows, columns=("year", "flow_id", "amount"))


def test_principal_inventory_preserves_exact_activation_order() -> None:
    """Keep severe opposing inventory terms with nonnegative demands."""
    bank = ProviderInventoryBank(
        inventories={
            ("providers", "providers", "a"): ProviderUnitInventory(
                flow_ids=np.array([9, 2, 5], dtype=np.int64),
                amounts=np.array([-971916599671.962, 0.0, 3.25], dtype=np.float64),
            ),
            ("providers", "providers", "b"): ProviderUnitInventory(
                flow_ids=np.array([5, 9, 1], dtype=np.int64),
                amounts=np.array([-2.0, -971916599671.962, 7.0], dtype=np.float64),
            ),
            # Opposing emissions and uptake use distinct inventories with positive provider demands.
            ("providers", "providers", "c"): ProviderUnitInventory(
                flow_ids=np.array([9, 2, 5], dtype=np.int64),
                amounts=np.array([971916599671.962, 0.0, -3.25], dtype=np.float64),
            ),
        },
        matrix_count=1,
        provider_count=3,
        solve_batch_count=1,
        unit_responses=pd.DataFrame(),
    )
    activations = pd.DataFrame(
        [
            {
                "year": 2050,
                "matrix_database": "providers",
                "provider_database": "providers",
                "provider_code": "c",
                "amount": 479015379216.0812,
            },
            {
                "year": 2040,
                "matrix_database": "providers",
                "provider_database": "providers",
                "provider_code": "b",
                "amount": -0.0,
            },
            {
                "year": 2050,
                "matrix_database": "providers",
                "provider_database": "providers",
                "provider_code": "b",
                "amount": 610055654026.0447,
            },
            {
                "year": 2050,
                "matrix_database": "providers",
                "provider_database": "providers",
                "provider_code": "a",
                "amount": 97398607671.17859,
            },
            {
                "year": 2040,
                "matrix_database": "providers",
                "provider_database": "providers",
                "provider_code": "c",
                "amount": 4.0,
            },
        ]
    )
    timeline = PrincipalTimeline(
        timeline=pd.DataFrame({"time": [2040, 2050]}),
        activations=activations,
        base_static_score=1.25,
        direct_inventory=pd.DataFrame(columns=("year", "flow_id", "amount")),
    )

    expected = _reference_inventory(timeline, bank)
    result = assemble_principal_inventory(timeline, bank)

    pd.testing.assert_frame_equal(result.inventory, expected, check_exact=True)
    assert result.timeline is timeline.timeline
    assert result.base_static_score == 1.25


def test_compact_timeline_preserves_principal_inventory_and_provider_order() -> None:
    bank = ProviderInventoryBank(
        inventories={
            ("providers_2025", "providers_2025", "b"): ProviderUnitInventory(
                flow_ids=np.array([2, 7], dtype=np.int64),
                amounts=np.array((-3.0, 4.0), dtype=np.float64),
            ),
            ("providers_2030", "providers_2030", "a"): ProviderUnitInventory(
                flow_ids=np.array([7, 9], dtype=np.int64),
                amounts=np.array((5.0, 6.0), dtype=np.float64),
            ),
        },
        matrix_count=2,
        provider_count=2,
        solve_batch_count=2,
        unit_responses=pd.DataFrame(),
    )
    activations = pd.DataFrame(
        [
            {
                "year": 2025,
                "matrix_database": "providers_2025",
                "provider_database": "providers_2025",
                "provider_code": "b",
                "provider_activity_id": 31,
                "provider_product_id": 41,
                "provider_service_id": "service_b",
                "provider_anchor_year": 2025,
                "amount": 2.0,
                "life_cycle_stage": "production",
                "ultimate_carbon_origin": "fossil",
            },
            {
                "year": 2030,
                "matrix_database": "providers_2030",
                "provider_database": "providers_2030",
                "provider_code": "a",
                "provider_activity_id": 32,
                "provider_product_id": 42,
                "provider_service_id": "service_a",
                "provider_anchor_year": 2030,
                "amount": 1.5,
                "life_cycle_stage": "recycling",
                "ultimate_carbon_origin": "biogenic",
            },
        ]
    )
    full = PrincipalTimeline(
        timeline=pd.DataFrame({"time": (2025, 2030)}),
        activations=activations,
        base_static_score=0.0,
        direct_inventory=pd.DataFrame({"year": [2026], "flow_id": [11], "amount": [0.125]}),
    )

    compact = compact_principal_timelines({"EUR": full})["EUR"]

    assert list(compact.activations.columns) == [
        "matrix_database",
        "provider_database",
        "provider_code",
        "provider_activity_id",
        "provider_product_id",
        "year",
        "amount",
    ]
    assert list(
        zip(
            compact.activations["provider_database"],
            compact.activations["provider_code"],
            strict=True,
        )
    ) == [("providers_2025", "b"), ("providers_2030", "a")]
    pd.testing.assert_frame_equal(
        assemble_principal_inventory(compact, bank).inventory,
        assemble_principal_inventory(full, bank).inventory,
        check_exact=True,
    )


def test_modelled_direct_flows_share_the_annual_inventory_and_method_switching() -> None:
    """Retain direct-only coordinates and combine native-unit quantities once.

    Invented flow 17 is m3 water; the other two nonzero flows are kg. These
    quantities cannot be combined across identities before their own CFs apply.
    No receiving activity is fabricated to characterize the direct exchanges.
    """
    bank = ProviderInventoryBank(
        inventories={
            ("matrix", "native", "supplier"): ProviderUnitInventory(
                flow_ids=np.array([5, 9], dtype=np.int64),
                amounts=np.array([-1.0, 5.0], dtype=np.float64),
            ),
        },
        matrix_count=1,
        provider_count=1,
        solve_batch_count=1,
        unit_responses=pd.DataFrame(),
    )
    timeline = PrincipalTimeline(
        timeline=pd.DataFrame(),
        activations=pd.DataFrame(
            {
                "matrix_database": ["matrix"],
                "provider_database": ["native"],
                "provider_code": ["supplier"],
                "year": [2025],
                "amount": [2.0],
            }
        ),
        base_static_score=0.0,
        direct_inventory=pd.DataFrame(
            {
                "year": [2025, 2025, 2025, 2025, 2026, 2026],
                "flow_id": [5, 9, 17, 17, 9, 25],
                "amount": [2.0, -3.0, 0.004, -0.001, 1.5, -0.5],
            }
        ),
    )
    principal = assemble_principal_inventory(timeline, bank)
    pd.testing.assert_frame_equal(
        principal.inventory,
        pd.DataFrame(
            {
                "year": [2025, 2025, 2026, 2026],
                "flow_id": [9, 17, 9, 25],
                "amount": [7.0, 0.003, 1.5, -0.5],
            }
        ),
        check_exact=True,
    )
    methods = tuple(
        ImpactCategory(index, "validation", "1", name, (name,), "impact unit", name)
        for index, name in enumerate(("first", "second"))
    )
    result = characterize_principal_inventory(
        principal,
        methods,
        CharacterizationFactorBank(
            factors={
                ("first",): {5: 3.0, 9: 2.0, 17: -4.0},
                ("second",): {9: -0.25, 17: 0.0, 25: 4.0},
            }
        ),
        foreground_scenario="validation",
        background_scenario="validation",
        background_dimensions={},
        region="EUR",
    )
    assert result.scores["score"].tolist() == pytest.approx([16.988, -4.125])
    annual = result.annual_scores.set_index(["impact_category_id", "year"])["score"]
    assert annual.loc[("first", 2025)] == pytest.approx(13.988)
    assert annual.loc[("first", 2026)] == 3.0
    assert annual.loc[("second", 2025)] == -1.75
    assert annual.loc[("second", 2026)] == -2.375
    assert annual.loc[("second", 2024)] == 0.0
