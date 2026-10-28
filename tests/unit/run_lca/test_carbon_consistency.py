import json
from pathlib import Path

import pandas as pd
import pytest

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.run_lca.calculation.carbon import (
    _stable_group_sum,
    carbon_consistency_table,
    direct_treatment_carbon_flows,
)


class _Node(dict[str, object]):
    def __init__(
        self,
        database: str,
        code: str,
        *,
        exchanges: tuple["_Exchange", ...] = (),
        production_amount: float = 1.0,
        **data: object,
    ) -> None:
        super().__init__(data)
        self.key = (database, code)
        self._exchanges = exchanges
        self._production_amount = production_amount

    def exchanges(self) -> tuple["_Exchange", ...]:
        return self._exchanges

    def rp_exchange(self) -> dict[str, float]:
        return {"amount": self._production_amount}


class _Exchange(dict[str, object]):
    def __init__(self, input_node: _Node, *, amount: float, exchange_type: str) -> None:
        super().__init__(amount=amount, type=exchange_type, input=input_node.key)
        self.input = input_node


class _Brightway:
    def __init__(self, nodes: tuple[_Node, ...]) -> None:
        self._nodes = {node.key: node for node in nodes}

    def Database(self, database: str) -> tuple[_Node, ...]:
        return tuple(node for key, node in self._nodes.items() if key[0] == database)


def _activation(*, year: int, amount: float, carbon: float) -> dict[str, object]:
    return {
        "year": year,
        "region": "EUR",
        "life_cycle_stage": "incineration_treatment",
        "material": "HDPE",
        "constituent": "hdpe",
        "product": "polymer",
        "virgin_feedstock_route": "all",
        "good": "packaging",
        "material_cycle_source": "not_applicable",
        "ultimate_carbon_origin": "fossil",
        "lca_fate": "incineration",
        "fate_origin": "remind_mfa",
        "provider_database": "frontier",
        "provider_code": "incineration_service",
        "provider_anchor_year": year,
        "amount": amount,
        "mfa_carbon_amount_kg": carbon,
    }


def test_stable_group_sum_bypasses_singletons_and_sums_only_repeated_rows() -> None:
    frame = pd.DataFrame(
        {
            "coordinate": [3, 1, 2, 2, 2],
            "amount": [3.0, 1.0, 1e16, 1.0, -1e16],
            "carbon": [0.3, 0.1, 1e16, 1.0, -1e16],
        }
    )

    result = _stable_group_sum(
        frame,
        ("coordinate",),
        ("amount", "carbon"),
    )

    assert result.to_dict(orient="records") == [
        {"coordinate": 1, "amount": 1.0, "carbon": 0.1},
        {"coordinate": 2, "amount": 1.0, "carbon": 1.0},
        {"coordinate": 3, "amount": 3.0, "carbon": 0.3},
    ]


def test_compact_carbon_diagnostic_uses_only_direct_native_treatment_carbon(
    tmp_path: Path,
) -> None:
    """Normalize positive native production once and exclude auxiliary supply-chain emissions."""
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="positive_treatment_production",
        biosphere_database="biosphere",
        source_database="ecoinvent",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": name,
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
                for code, name in (
                    ("co2", "Carbon dioxide, fossil"),
                    ("methane", "Methane, fossil"),
                    ("benzene", "Benzene"),
                )
            },
            searchable=False,
        )
        bd.Database("ecoinvent").write(
            {
                ("ecoinvent", code): {
                    "name": code,
                    "reference product": "treatment operation",
                    "unit": "kilogram",
                    "location": "RER",
                    "type": "process",
                    "exchanges": [
                        {"input": ("ecoinvent", code), "amount": production, "type": "production"},
                        *[
                            {"input": ("biosphere", flow), "amount": amount, "type": "biosphere"}
                            for flow, amount in emissions
                        ],
                    ],
                }
                for code, production, emissions in (
                    (
                        "native_incineration",
                        2.0,
                        (("co2", 4.4), ("methane", 0.16), ("benzene", 10.0)),
                    ),
                    ("auxiliary", 1.0, (("co2", 100.0),)),
                )
            },
            searchable=False,
        )
        bd.Database("frontier").write(
            {
                ("frontier", "incineration_service"): {
                    "name": "Incineration service",
                    "reference product": "waste treatment service",
                    "unit": "kilogram",
                    "location": "RER",
                    "type": "process",
                    "direct_treatment_carbon_owners_json": json.dumps(
                        [
                            {
                                "owner_database": "ecoinvent",
                                "owner_code": "native_incineration",
                                "owner_scale_per_provider_unit": 0.5,
                                "owner_scope": "direct_biosphere_only",
                                "source_role": "native_ecoinvent_provider",
                                "origin_mapping_status": "native_identity",
                            }
                        ]
                    ),
                    "exchanges": [
                        {
                            "input": ("frontier", "incineration_service"),
                            "amount": 1.0,
                            "type": "production",
                        },
                        {
                            "input": ("ecoinvent", "native_incineration"),
                            "amount": 0.5,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ecoinvent", "auxiliary"),
                            "amount": 1.0,
                            "type": "technosphere",
                        },
                    ],
                }
            },
            searchable=False,
        )
        activations = pd.DataFrame(
            [
                _activation(year=2050, amount=1.0, carbon=0.5),
                _activation(year=2050, amount=0.5, carbon=0.25),
                {
                    **_activation(year=2050, amount=0.5, carbon=0.2),
                    "material": "BioPlastics",
                    "constituent": "other_biodegradable",
                },
                {
                    **_activation(year=2050, amount=0.75, carbon=0.1),
                    "material": "BioPlastics",
                    "constituent": "starch_blends",
                },
            ]
        )
        result = carbon_consistency_table(bd, activations)
        flows = direct_treatment_carbon_flows(bd, activations)

    assert len(result) == 3
    row = result.loc[result.constituent.eq("hdpe")].iloc[0]
    assert result.set_index("constituent").mfa_treatment_input_carbon_kg.to_dict() == {
        "hdpe": 0.75,
        "other_biodegradable": 0.2,
        "starch_blends": 0.1,
    }
    assert result.mfa_carbon_boundary.eq("treatment_gate_constituent_carbon").all()
    assert flows.constituent.nunique() == 3
    assert flows.carbon_amount.sum() == pytest.approx(2.75 * 0.33)
    assert row["provider_demand_amount"] == pytest.approx(1.5)
    assert row["mfa_treatment_input_carbon_kg"] == pytest.approx(0.75)
    assert row["native_direct_co2_carbon_kg"] == pytest.approx(0.45)
    assert row["native_direct_ch4_carbon_kg"] == pytest.approx(0.045)
    assert row["native_direct_co_carbon_kg"] == 0.0
    assert row["native_direct_selected_carbon_kg"] == pytest.approx(0.495)
    assert json.loads(row["native_direct_carbon_flow_keys_json"]) == [
        ["biosphere", "co2"],
        ["biosphere", "methane"],
    ]
    assert row["comparison_additivity"] == "non_additive_custody_diagnostic"
    assert flows.set_index("carbon_species")[
        "flow_amount_per_provider_unit"
    ].to_dict() == pytest.approx({"carbon dioxide": 1.1, "methane": 0.04})


def test_generated_treatment_reads_only_its_direct_self_owned_carbon() -> None:
    carbon_dioxide = _Node(
        "biosphere",
        "co2_non_fossil",
        name="Carbon dioxide, non-fossil",
        unit="kilogram",
    )
    auxiliary_carbon_dioxide = _Node(
        "biosphere",
        "co2_fossil",
        name="Carbon dioxide, fossil",
        unit="kilogram",
    )
    auxiliary = _Node(
        "ecoinvent",
        "auxiliary",
        exchanges=(
            _Exchange(
                auxiliary_carbon_dioxide,
                amount=44.0,
                exchange_type="biosphere",
            ),
        ),
        name="Carbon-emitting auxiliary service",
        **{
            "reference product": "auxiliary service",
            "unit": "kilogram",
            "location": "RER",
        },
    )
    provider = _Node(
        "frontier",
        "incineration_non_fossil",
        exchanges=(
            _Exchange(carbon_dioxide, amount=4.4, exchange_type="biosphere"),
            _Exchange(auxiliary, amount=1.0, exchange_type="technosphere"),
        ),
        production_amount=2.0,
        name="Non-fossil incineration service",
        **{
            "reference product": "waste treatment service",
            "unit": "kilogram",
            "location": "RER",
            "direct_treatment_carbon_owners_json": json.dumps(
                [
                    {
                        "owner_database": "frontier",
                        "owner_code": "incineration_non_fossil",
                        "owner_scale_per_provider_unit": 1.0,
                        "owner_scope": "direct_biosphere_only",
                        "source_role": "member_local_frontier_service",
                        "origin_mapping_status": "exact_non_fossil_counterpart",
                    }
                ]
            ),
        },
    )
    activation = _activation(year=2050, amount=2.0, carbon=1.0)
    activation["ultimate_carbon_origin"] = "biogenic"
    activation["provider_code"] = "incineration_non_fossil"

    brightway = _Brightway((provider, auxiliary, carbon_dioxide, auxiliary_carbon_dioxide))
    result = carbon_consistency_table(brightway, pd.DataFrame([activation]))

    row = result.iloc[0]
    assert row["native_direct_co2_carbon_kg"] == pytest.approx(1.2)
    assert row["native_direct_ch4_carbon_kg"] == 0.0
    assert json.loads(row["native_direct_carbon_flow_keys_json"]) == [
        ["biosphere", "co2_non_fossil"]
    ]


def test_composite_treatment_reports_every_declared_direct_carbon_owner() -> None:
    carbon_dioxide = _Node(
        "biosphere",
        "co2",
        name="Carbon dioxide, fossil",
        unit="kilogram",
        categories=("air",),
    )
    methane = _Node(
        "biosphere",
        "methane",
        name="Methane, non-fossil",
        unit="kilogram",
        categories=("air",),
    )
    carbon_monoxide = _Node(
        "biosphere",
        "co",
        name="Carbon monoxide",
        unit="kilogram",
        categories=("air",),
    )
    conversion = _Node(
        "native",
        "conversion",
        exchanges=(
            _Exchange(carbon_dioxide, amount=4.4, exchange_type="biosphere"),
            _Exchange(carbon_monoxide, amount=0.28, exchange_type="biosphere"),
        ),
        production_amount=2.0,
        name="Hydrothermal conversion",
    )
    reject_treatment = _Node(
        "native",
        "reject_treatment",
        exchanges=(_Exchange(methane, amount=0.16, exchange_type="biosphere"),),
        name="Reject treatment",
    )
    owners = [
        {
            "owner_database": "native",
            "owner_code": "conversion",
            "owner_scale_per_provider_unit": 0.5,
            "owner_scope": "direct_biosphere_only",
            "source_role": "conversion_operation",
            "origin_mapping_status": "native_identity",
        },
        {
            "owner_database": "native",
            "owner_code": "reject_treatment",
            "owner_scale_per_provider_unit": 0.25,
            "owner_scope": "direct_biosphere_only",
            "source_role": "preparation_reject_treatment",
            "origin_mapping_status": "exact_non_fossil_counterpart",
        },
    ]
    provider = _Node(
        "frontier",
        "chemical_service",
        name="Chemical recycling service",
        **{
            "direct_treatment_carbon_owners_json": json.dumps(owners),
            "reference product": "chemical recycling service",
            "unit": "kilogram",
            "location": "GB",
        },
    )
    activation = _activation(year=2050, amount=2.0, carbon=1.0)
    activation.update(
        {
            "life_cycle_stage": "chemical_recycling_preparation",
            "provider_code": "chemical_service",
            "provider_anchor_year": 2050,
            "material_cycle_source": "chemically_recycled_feedstock",
        }
    )
    nodes = (
        provider,
        conversion,
        reject_treatment,
        carbon_dioxide,
        methane,
        carbon_monoxide,
    )

    flows = direct_treatment_carbon_flows(
        _Brightway(nodes),
        pd.DataFrame([activation]),
    )
    assert flows[["owner_code", "carbon_species"]].values.tolist() == [
        ["conversion", "carbon dioxide"],
        ["conversion", "carbon monoxide"],
        ["reject_treatment", "methane"],
    ]
    assert flows["carbon_amount"].tolist() == pytest.approx([0.6, 0.06, 0.06])
    assert flows["flow_carbon_origin"].tolist() == ["fossil", "unspecified", "non_fossil"]
    assert flows["provider_anchor_year"].tolist() == [2050, 2050, 2050]

    compact = carbon_consistency_table(_Brightway(nodes), pd.DataFrame([activation]))
    assert compact.iloc[0]["native_direct_selected_carbon_kg"] == pytest.approx(0.72)
    assert compact.iloc[0]["mfa_carbon_boundary"] == "treatment_gate_constituent_carbon"
    activation["life_cycle_stage"] = "chemical_recycling_conversion"
    activation["mfa_carbon_amount_kg"] = 0.8
    processed = carbon_consistency_table(_Brightway(nodes), pd.DataFrame([activation]))
    assert processed.iloc[0]["mfa_treatment_input_carbon_kg"] == 0.8
    assert (
        processed.iloc[0]["mfa_carbon_boundary"] == "processed_constituent_carbon_after_preparation"
    )


def test_non_treatment_activations_are_outside_the_diagnostic() -> None:
    provider = _Node(
        "frontier",
        "incineration_service",
        name="Incineration service",
        **{
            "reference product": "waste treatment service",
            "unit": "kilogram",
            "location": "RER",
        },
    )
    activation = _activation(year=2050, amount=1.0, carbon=0.5)
    activation["life_cycle_stage"] = "virgin_resin_production"

    result = carbon_consistency_table(_Brightway((provider,)), pd.DataFrame([activation]))

    assert result.empty
    assert direct_treatment_carbon_flows(_Brightway((provider,)), pd.DataFrame([activation])).empty


def test_treatment_without_direct_carbon_flows_returns_an_empty_flow_table() -> None:
    wrong_unit = _Node(
        "biosphere",
        "co2_moles",
        name="Carbon dioxide, fossil",
        unit="mole",
    )
    other_flow = _Node(
        "biosphere",
        "benzene",
        name="Benzene",
        unit="kilogram",
    )
    auxiliary = _Node("native", "auxiliary", name="Auxiliary process")
    owners = [
        {
            "owner_database": "frontier",
            "owner_code": "incineration_service",
            "owner_scale_per_provider_unit": 1.0,
            "owner_scope": "direct_biosphere_only",
            "source_role": "member_local_frontier_service",
            "origin_mapping_status": "native_identity",
        }
    ]
    provider = _Node(
        "frontier",
        "incineration_service",
        exchanges=(
            _Exchange(auxiliary, amount=1.0, exchange_type="technosphere"),
            _Exchange(wrong_unit, amount=1.0, exchange_type="biosphere"),
            _Exchange(other_flow, amount=1.0, exchange_type="biosphere"),
        ),
        name="Incineration service",
        **{
            "reference product": "waste treatment service",
            "unit": "kilogram",
            "location": "RER",
            "direct_treatment_carbon_owners_json": json.dumps(owners),
        },
    )
    activation = pd.DataFrame([_activation(year=2050, amount=1.0, carbon=0.5)])
    brightway = _Brightway((provider, auxiliary, wrong_unit, other_flow))

    flows = direct_treatment_carbon_flows(brightway, activation)
    consistency = carbon_consistency_table(brightway, activation)

    assert flows.empty
    assert {"flow_name", "flow_amount", "carbon_amount"} <= set(flows.columns)
    assert consistency.iloc[0]["native_direct_selected_carbon_kg"] == 0.0
