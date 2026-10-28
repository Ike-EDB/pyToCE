"""Installed Brightway and bw_timex boundary checks for principal inventories."""

from math import isfinite, log
from pathlib import Path
from sys import float_info

import pandas as pd
import pytest
from test_contribution_accumulation import _activation

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    create_contribution_state,
    create_provider_response_state,
)
from pytoce.run_lca.calculation.timex import (
    PrincipalTimeline,
    assemble_principal_inventory,
    build_principal_timelines,
    build_provider_inventory_bank,
)

# The existing chain uses finite exchanges representable in Brightway's float32 data.
_CHAIN_REQUIREMENT = 1e30
_FINITE_CHAIN_STEPS = int(log(float_info.max) / log(_CHAIN_REQUIREMENT))
_HIGH_FINITE_FACTOR = 10.0 ** int(log(float_info.max / _CHAIN_REQUIREMENT**_FINITE_CHAIN_STEPS, 10))


def _names(tmp_path: Path, project: str) -> BrightwayNames:
    """Return isolated Brightway names for one compact integration fixture."""

    return BrightwayNames(
        store=tmp_path / "brightway",
        project=project,
        biosphere_database="biosphere",
        source_database="source",
    )


def test_principal_timeline_rejects_a_direct_regional_root_burden(tmp_path: Path) -> None:
    """Keep regional roots free of burdens owned by dated provider activities."""

    names = _names(tmp_path, "direct_root_burden")
    method = ("compact method", "climate change")
    with selected_brightway_project(names) as bd:
        bd.geomapping.add(("EUR",))
        bd.Database(names.biosphere_database).write(
            {
                (names.biosphere_database, "co2"): {
                    "name": "carbon dioxide",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                }
            },
            searchable=False,
        )
        bd.Database("foreground").write(
            {
                ("foreground", "root_eur"): {
                    "name": "regional plastics system, EUR",
                    "reference product": "regional plastics system",
                    "unit": "unit",
                    "location": "EUR",
                    "region": "EUR",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": ("foreground", "root_eur"),
                            "amount": 1.0,
                            "type": "production",
                        },
                        {
                            "input": (names.biosphere_database, "co2"),
                            "amount": 1.0,
                            "type": "biosphere",
                        },
                    ],
                }
            },
            searchable=False,
        )
        bd.Method(method).register(unit="kilogram carbon dioxide equivalent")
        bd.Method(method).write([((names.biosphere_database, "co2"), 1.0)])
        root = bd.get_node(database="foreground", code="root_eur")
        activations = pd.DataFrame(columns=["region", "year", "provider_service_code"])

        with pytest.raises(ValueError, match="Timex returned a non-zero score"):
            build_principal_timelines(
                {"EUR": root},
                method,
                {"foreground": "dynamic"},
                bd,
                activations,
                {},
                {},
                use_global_lci_cache=False,
            )


@pytest.mark.parametrize(
    (
        "inventory_amount",
        "factor",
        "supply_steps",
        "cancel_inventory",
        "activation_amount",
        "assemble_inventory",
        "expected_score",
        "error",
    ),
    [
        (-2.0, 3.0, 0, False, None, False, -6.0, None),
        (1e39, 3.0, 0, False, None, False, None, "provider_2025.*nonfinite inventory.*service"),
        (2.0, -3.0, 0, False, None, False, -6.0, None),
        (2.0, 0.0, 0, False, None, False, 0.0, None),
        (
            2.0,
            1e39,
            0,
            False,
            None,
            False,
            None,
            "compact method.*nonfinite characterization factors.*provider_2025",
        ),
        pytest.param(
            1.0,
            1.0,
            12,
            False,
            None,
            False,
            None,
            "provider_2025.*nonfinite supply.*node11.*node12",
            id="finite_chain_supply_overflow",
        ),
        pytest.param(
            1.0,
            _CHAIN_REQUIREMENT,
            _FINITE_CHAIN_STEPS,
            False,
            None,
            False,
            None,
            rf"(?=.*provider_2025)(?=.*node{_FINITE_CHAIN_STEPS:02d})"
            r"(?=.*compact method).*nonfinite",
            id="finite_chain_score_overflow",
        ),
        pytest.param(
            1.0,
            _CHAIN_REQUIREMENT,
            _FINITE_CHAIN_STEPS,
            True,
            None,
            False,
            None,
            rf"(?=.*provider_2025)(?=.*node{_FINITE_CHAIN_STEPS:02d})"
            r"(?=.*compact method).*nonfinite",
            id="finite_chain_norm_overflow_after_cancellation",
        ),
        pytest.param(
            1.0,
            _HIGH_FINITE_FACTOR,
            _FINITE_CHAIN_STEPS,
            False,
            2.0,
            False,
            None,
            rf"(?=.*provider_2025)(?=.*node{_FINITE_CHAIN_STEPS:02d})"
            r"(?=.*compact method).*nonfinite",
            id="finite_unit_response_activation_overflow",
        ),
        pytest.param(
            _HIGH_FINITE_FACTOR,
            0.0,
            _FINITE_CHAIN_STEPS,
            False,
            2.0,
            True,
            None,
            rf"(?=.*2050)(?=.*provider_2025)(?=.*node{_FINITE_CHAIN_STEPS:02d}).*nonfinite.*inventory",
            id="zero_factor_finite_inventory_activation_overflow",
        ),
    ],
)
def test_provider_bank_admits_finite_inventory(
    tmp_path: Path,
    inventory_amount: float,
    factor: float,
    supply_steps: int,
    cancel_inventory: bool,
    activation_amount: float | None,
    assemble_inventory: bool,
    expected_score: float | None,
    error: str | None,
) -> None:
    """Reject nonfinite supplies, inventories, factors or responses; preserve signed impacts."""

    names = _names(tmp_path, "provider_inventory_admission")
    with selected_brightway_project(names) as bd:
        bd.Database(names.biosphere_database).write(
            {
                (names.biosphere_database, "emission"): {
                    "name": "test emission",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                }
            },
            searchable=False,
        )
        data = {
            ("provider_2025", "service"): {
                "name": "provider service",
                "reference product": "provider service",
                "unit": "kilogram",
                "location": "GLO",
                "type": "process",
                "exchanges": [
                    {
                        "input": ("provider_2025", "service"),
                        "amount": 1.0,
                        "type": "production",
                    },
                    {
                        "input": (names.biosphere_database, "emission"),
                        "amount": inventory_amount,
                        "type": "biosphere",
                    },
                ],
            }
        }
        if cancel_inventory:
            # Equal signed emissions from distinct suppliers cancel in the solved
            # inventory, while their absolute activity contributions still add.
            data[("provider_2025", "offset")] = {
                "name": "offsetting supplier",
                "reference product": "offsetting supplier",
                "unit": "kilogram",
                "location": "GLO",
                "type": "process",
                "exchanges": [
                    {"input": ("provider_2025", "offset"), "amount": 1.0, "type": "production"},
                    {
                        "input": (names.biosphere_database, "emission"),
                        "amount": -inventory_amount,
                        "type": "biosphere",
                    },
                ],
            }
        previous = "service"
        for step in range(1, supply_steps + 1):
            code = f"node{step:02d}"
            # Each native exchange is finite. Repeated requirements can exceed
            # float64 during supply calculation or subsequent characterization.
            data[("provider_2025", code)] = {
                "name": code,
                "reference product": code,
                "unit": "kilogram",
                "location": "GLO",
                "type": "process",
                "exchanges": [
                    {"input": ("provider_2025", code), "amount": 1.0, "type": "production"},
                    {
                        "input": ("provider_2025", previous),
                        "amount": _CHAIN_REQUIREMENT,
                        "type": "technosphere",
                    },
                ],
            }
            if cancel_inventory and step == 1:
                data[("provider_2025", code)]["exchanges"].append(
                    {
                        "input": ("provider_2025", "offset"),
                        "amount": _CHAIN_REQUIREMENT,
                        "type": "technosphere",
                    }
                )
            previous = code
        bd.Database("provider_2025").write(data, searchable=False)
        assert all(
            isfinite(exchange["amount"])
            for node in bd.Database("provider_2025")
            for exchange in node.exchanges()
        )
        flow = bd.get_node(database=names.biosphere_database, code="emission")
        provider = bd.get_node(database="provider_2025", code="service")
        method = ContributionMethod(
            method_id="compact_method",
            method_version="1",
            impact_category_id="compact_category",
            method_tuple=("compact method", "compact category"),
            unit="impact unit",
        )
        bd.Method(method.method_tuple).register(unit=method.unit)
        bd.Method(method.method_tuple).write([(flow.key, factor)])
        endpoints = (
            [
                bd.get_node(database="provider_2025", code=f"node{step:02d}")
                for step in (supply_steps - 1, supply_steps)
            ]
            if supply_steps
            else [provider]
        )
        requests = pd.DataFrame(
            [
                {
                    "matrix_database": "provider_2025",
                    "provider_database": "provider_2025",
                    "provider_code": endpoint["code"],
                    "provider_activity_id": int(endpoint.id),
                    "provider_product_id": int(endpoint.rp_exchange().input.id),
                    "provider_anchor_year": 2025,
                }
                for endpoint in endpoints
            ]
        )
        if activation_amount is None:
            state = create_provider_response_state(bd, (method,))
        else:
            # Unit responses remain finite; requesting two reference units can
            # exceed float64 when the completed foreground amount is applied.
            activations = pd.DataFrame(
                [
                    {**_activation("provider_2025"), **row, "amount": activation_amount}
                    for row in requests.to_dict(orient="records")
                ]
            )
            state = create_contribution_state(
                bd, {"EUR": activations}, {"EUR": provider.key}, (method,), (2050,)
            )
        if error is not None:
            if assemble_inventory:
                bank = build_provider_inventory_bank(bd, requests, state)
                assert state.annual_totals[("EUR", 2050, 0)] == (0.0, 0.0)
                timeline = PrincipalTimeline(
                    pd.DataFrame(),
                    activations,
                    0.0,
                    pd.DataFrame(columns=("year", "flow_id", "amount")),
                )
                with pytest.raises(ValueError, match=error) as failure:
                    assemble_principal_inventory(timeline, bank)
                assert str(flow.id) in str(failure.value)
            else:
                with pytest.raises(ValueError, match=error):
                    bank = build_provider_inventory_bank(bd, requests, state)
            assert next(iter(provider.biosphere()))["amount"] == inventory_amount
            assert bd.Method(method.method_tuple).load() == [(flow.id, factor)]
            return

        bank = build_provider_inventory_bank(bd, requests, state)

    assert bank.matrix_count == 1
    assert bank.provider_count == 1
    assert bank.solve_batch_count == 1
    assert bank.unit_responses.iloc[0]["unit_score"] == expected_score
    assert expected_score is not None
    assert bank.unit_responses.iloc[0]["unit_contribution_norm"] == abs(expected_score)


@pytest.mark.parametrize("retain_contributions", [False, True])
def test_native_product_uses_selected_matrix_and_signed_production(
    tmp_path: Path, retain_contributions: bool
) -> None:
    """Solve one real native endpoint in two different complete upstream contexts."""

    names = _names(tmp_path, "native_product_matrix_context")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "emission"): {
                    "name": "test emission",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                }
            },
            searchable=False,
        )
        bd.Database("products").write(
            {
                ("products", "feed"): {
                    "name": "feed",
                    "unit": "kilogram",
                    "type": "product",
                }
            },
            searchable=False,
        )
        bd.Database("native").write(
            {
                ("native", "waste"): {
                    "name": "waste",
                    "unit": "kilogram",
                    "type": "product",
                },
                ("native", "receiver"): {
                    "name": "native waste receiver",
                    "reference product": "waste",
                    "unit": "kilogram",
                    "location": "GLO",
                    "type": "process",
                    "exchanges": [
                        {"input": ("native", "waste"), "type": "production", "amount": -2},
                        {"input": ("products", "feed"), "type": "technosphere", "amount": 3},
                        {"input": ("biosphere", "emission"), "type": "biosphere", "amount": -1},
                    ],
                },
            },
            searchable=False,
        )
        for matrix, emission in (("matrix_a", 4), ("matrix_b", 10)):
            bd.Database(matrix).write(
                {
                    (matrix, "producer"): {
                        "name": "selected feed producer",
                        "reference product": "feed",
                        "unit": "kilogram",
                        "location": "GLO",
                        "type": "process",
                        "exchanges": [
                            {"input": ("products", "feed"), "type": "production", "amount": 1},
                            {
                                "input": ("biosphere", "emission"),
                                "type": "biosphere",
                                "amount": emission,
                            },
                        ],
                    }
                },
                searchable=False,
            )
        methods = tuple(
            ContributionMethod(
                method_id=label,
                method_version="1",
                impact_category_id="test",
                method_tuple=(label, "test"),
                unit="impact unit",
            )
            for label in ("positive", "negative")
        )
        flow = bd.get_node(database="biosphere", code="emission")
        for method, factor in zip(methods, (2, -3), strict=True):
            bd.Method(method.method_tuple).register(unit=method.unit)
            bd.Method(method.method_tuple).write([(flow.key, factor)])
        receiver = bd.get_node(database="native", code="receiver")
        product = receiver.rp_exchange().input
        assert product.id != receiver.id
        frames = {}
        for region, matrix, anchor, amount in (
            ("EUR", "matrix_a", 2025, -4),
            ("USA", "matrix_b", 2050, -6),
        ):
            frames[region] = pd.DataFrame(
                [
                    {
                        **_activation(matrix),
                        "provider_database": "native",
                        "provider_code": "receiver",
                        "provider_activity_id": receiver.id,
                        "provider_product_id": product.id,
                        "provider_anchor_year": anchor,
                        "amount": amount,
                    }
                ]
            )
        state = (
            create_contribution_state(
                bd,
                frames,
                {region: receiver.key for region in frames},
                methods,
                (2050,),
            )
            if retain_contributions
            else create_provider_response_state(bd, methods)
        )
        # A repeated request is one solve; the same native key in another matrix
        # is a distinct inventory because its selected upstream producer differs.
        requests = pd.concat([frames["EUR"], frames["USA"], frames["EUR"]], ignore_index=True)
        bank = build_provider_inventory_bank(bd, requests, state)
        assert (bank.matrix_count, bank.provider_count, bank.solve_batch_count) == (2, 2, 2)
        # Native solver roundoff can affect solved amounts and their derived scores.
        if retain_contributions:
            assert bank.unit_responses.empty
            for region, matrix, unit_amount, expected_inventory, score, norm in (
                ("EUR", "matrix_a", -5.5, 22, 44, 52),
                ("USA", "matrix_b", -14.5, 87, 174, 186),
            ):
                inventory = bank.inventories[(matrix, "native", "receiver")]
                assert inventory.flow_ids.tolist() == [flow.id]
                assert inventory.amounts.tolist() == pytest.approx(
                    [unit_amount], rel=1e-12, abs=1e-14
                )
                timeline = PrincipalTimeline(
                    pd.DataFrame(),
                    frames[region],
                    0.0,
                    pd.DataFrame(columns=("year", "flow_id", "amount")),
                )
                assembled = assemble_principal_inventory(timeline, bank)
                assert assembled.inventory["amount"].tolist() == pytest.approx(
                    [expected_inventory], rel=1e-12, abs=1e-14
                )
                assert state.annual_totals[(region, 2050, 0)] == pytest.approx(
                    (score, norm), rel=1e-12, abs=1e-14
                )
                assert state.annual_totals[(region, 2050, 1)] == pytest.approx(
                    (-1.5 * score, 1.5 * norm), rel=1e-12, abs=1e-14
                )
        else:
            assert not bank.inventories
            assert "provider_service_id" not in bank.unit_responses.columns
            for row in bank.unit_responses.itertuples(index=False):
                unit, norm = (-5.5, 6.5) if row.matrix_database == "matrix_a" else (-14.5, 15.5)
                factor = 2 if row.method_id == "positive" else -3
                assert (row.provider_database, row.provider_code) == receiver.key
                assert row.provider_activity_id == receiver.id
                assert row.provider_product_id == product.id
                assert (row.provider_product_database, row.provider_product_code) == product.key
                assert row.provider_unit == "kilogram"
                assert row.unit_score == pytest.approx(unit * factor, rel=1e-12, abs=1e-14)
                assert row.unit_contribution_norm == pytest.approx(
                    norm * abs(factor), rel=1e-12, abs=1e-14
                )
