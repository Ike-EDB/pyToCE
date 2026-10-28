"""Explain how MFA quantities enter, or are already represented in, LCA."""

import json
from dataclasses import dataclass
from math import fsum
from typing import cast

import pandas as pd

from pytoce.run_mfa.model.mfa import MFAEvaluation

from .burdens import BURDEN_SPECS

FLOW_OWNERSHIP_METHOD_IDENTITY = "pytoce_constituent_flow_ownership_v3"
_SPECS = {spec.demand_type: spec for spec in BURDEN_SPECS}


@dataclass(frozen=True, slots=True)
class FlowUse:
    """State a quantity's role and related burdens."""

    classification: str
    demands: tuple[str, ...]
    reason: str
    constituent_axis: str = "source"


_NEW_POLYMER = FlowUse(
    "direct_lca_input",
    (
        "virgin_resin_production",
        "polymerization_operation",
        "virgin_monomer_supply",
        "recovered_monomer_supply",
    ),
    "Domestic and exported output together define D at the producing region/year. "
    "Noneligible constituents use virgin resin; eligible constituents use one "
    "polymerization operation and independent ethylene/propylene supply fractions.",
    "destination",
)

_MATERIAL_USES = {
    "collected_after_trade": FlowUse(
        "custody_transfer",
        (),
        "Collected waste plus imports minus exports defines available treatment input; "
        "the partitioned treatments own the LCA demands.",
    ),
    "mechanical_treatment": FlowUse(
        "direct_lca_input",
        ("mechanical_recycling_operation",),
        "Charge mechanical recycling on its actual constituent input.",
    ),
    "chemical_treatment": FlowUse(
        "direct_lca_input",
        ("chemical_recycling_preparation", "chemical_recycling_conversion"),
        "Preparation and hydrothermal conversion both use Q, the tracked plastic "
        "entering preparation. The conversion inventory already includes P/Q normalization.",
    ),
    "incineration_treatment": FlowUse(
        "direct_lca_input",
        ("incineration_treatment",),
        "Charge terminal incineration on its actual constituent waste input.",
    ),
    "landfill_treatment": FlowUse(
        "direct_lca_input",
        ("landfill_treatment",),
        "Charge terminal landfill on its actual constituent waste input.",
    ),
    "mechanical_resin": FlowUse(
        "already_owned",
        ("mechanical_recycling_operation",),
        "The recycling input already owns resin recovery. Recovered polymer enters "
        "subsequent processing.",
    ),
    "mechanical_landfilled_reject": FlowUse(
        "direct_lca_input",
        ("mechanical_recycling_reject_landfill_treatment",),
        "Charge the source-specific managed landfill fate on its actual principal reject.",
    ),
    "mechanical_waterborne_loss": FlowUse(
        "physical_process_output",
        ("mechanical_recycling_operation",),
        "Principal entering wastewater is separate from dry managed reject. The recovery "
        "operation includes wastewater treatment.",
    ),
    "mechanical_airborne_loss": FlowUse(
        "already_owned",
        ("mechanical_recycling_operation",),
        "Native principal dust belongs once to the recovery operation and is subtracted "
        "from its dry managed reject.",
    ),
    "mechanical_incinerated_reject": FlowUse(
        "direct_lca_input",
        ("mechanical_recycling_reject_incineration_treatment",),
        "Charge the separately modeled mechanical reject to incineration.",
    ),
    "chemical_processed_feed": FlowUse(
        "already_owned",
        ("chemical_recycling_conversion",),
        "P = Q times the native processed fraction. This polymer-account exit "
        "and carbon input is already represented by the Q-normalized conversion service.",
    ),
    "chemical_preparation_reject": FlowUse(
        "already_owned",
        ("chemical_recycling_preparation",),
        "Q - P leaves the polymer account; its treatment belongs to the native "
        "preparation/reject chain, with no additional generic treatment demand.",
    ),
    "new_polymer_domestic": _NEW_POLYMER,
    "new_polymer_export": _NEW_POLYMER,
    "manufacturing_principal_input": FlowUse(
        "already_owned",
        _NEW_POLYMER.demands + ("mechanical_recycling_operation",),
        "Gross principal demand is supplied by new, traded or recovered resin. "
        "Those suppliers already own their production; no second resin purchase is made.",
    ),
    "manufacturing_finished_output": FlowUse(
        "already_owned",
        ("processing_conversion",),
        "Finished principal partitions the conversion portfolio. Its aggregate domestic "
        "and exported views drive the same single conversion demand.",
    ),
    "manufacturing_principal_loss": FlowUse(
        "already_owned",
        ("processing_conversion",),
        "Actual source-specific conversion losses are inside gross principal supply "
        "and their conversion operation. Named releases and residual treatment belong "
        "to that operation, without a second generic waste demand.",
    ),
    "hydrothermal_product": FlowUse(
        "physical_process_output",
        ("chemical_recycling_conversion",),
        "Gross naphtha and distillate gas oil feed the cracker calculation; heavy gas "
        "oil and wax remain nonplastic outputs. These are not additional inventory "
        "purchases, disposal demands or avoided-product credits.",
    ),
    "cracker_product": FlowUse(
        "physical_process_output",
        ("recovered_monomer_supply",),
        "Gross ethylene/propylene supply regional recovered pools. Other coproducts "
        "retain their reported amounts as nonplastic coproducts. "
        "Allocated cracker burdens belong to monomer services.",
    ),
    "recovered_monomer_consumption": FlowUse(
        "monomer_allocation_input",
        ("recovered_monomer_supply",),
        "R is Mt monomer for a destination constituent. Each E/P service "
        "uses D * R/(N+R) kg destination polymer after converting D from Mt to kg; "
        "regional origin-traced pools provide the retained carbon.",
        "destination",
    ),
    "virgin_monomer_consumption": FlowUse(
        "monomer_allocation_input",
        ("virgin_monomer_supply",),
        "N is Mt monomer for a destination constituent. Each E/P service "
        "uses D * N/(N+R) * virgin_route_share kg destination polymer after converting "
        "D from Mt to kg.",
        "destination",
    ),
}

_FLOW_USES = {
    **{
        flow: FlowUse(
            "aggregate_projection",
            _MATERIAL_USES[material_flow].demands,
            f"Aggregate view of material_realization.parquet:{material_flow}; "
            "the constituent quantity owns its LCA use. Do not charge this projection again.",
            "not_applicable",
        )
        for flow, material_flow in (
            ("new_polymer => processing", "new_polymer_domestic"),
            ("new_polymer => primary_market", "new_polymer_export"),
            ("collected => reclmech", "mechanical_treatment"),
            ("collected => reclchem", "chemical_treatment"),
            ("collected => incineration", "incineration_treatment"),
            ("collected => landfill", "landfill_treatment"),
            ("reclmech => processing", "mechanical_resin"),
            ("reclmech => landfill", "mechanical_landfilled_reject"),
            ("reclmech => waterborne_loss", "mechanical_waterborne_loss"),
            ("reclmech => airborne_loss", "mechanical_airborne_loss"),
            ("reclmech => incineration", "mechanical_incinerated_reject"),
            ("processing => manufacturing_loss", "manufacturing_principal_loss"),
            ("reclchem => chemical_feedstock", "chemical_processed_feed"),
            ("reclchem => chemical_preparation_reject", "chemical_preparation_reject"),
        )
    },
    **{
        flow: FlowUse(
            "direct_lca_input",
            ("processing_conversion",),
            "Sum domestic and exported processed plastic once at the producing "
            "region/year, then partition finished output by constituent.",
            "not_applicable",
        )
        for flow in ("processing => fabrication", "processing => intermediate_market")
    },
    "eol => mismanaged": FlowUse(
        "direct_lca_input",
        ("mismanaged_open_burning_treatment", "mismanaged_open_dump_treatment"),
        "Partition this family input once by constituent shares and the common "
        "burning/dumping shares applied to every world region, then charge the resulting "
        "waste treatments.",
        "not_applicable",
    ),
    "intermediate_market => fabrication": FlowUse(
        "already_owned",
        ("processing_conversion",),
        "Receipt of traded processed material; conversion belongs to the producing region.",
        "not_applicable",
    ),
    "primary_market => processing": FlowUse(
        "already_owned",
        _NEW_POLYMER.demands,
        "Receipt of traded new polymer; production belongs to source-region exports.",
        "not_applicable",
    ),
    **{
        flow: FlowUse(
            "custody_transfer",
            (),
            "Stock, collection or trade custody. No additional production, "
            "conversion, transport or treatment is inferred from this transfer alone. "
            "Stock still in use at end-2100 receives no artificial discard or burden.",
            "not_applicable",
        )
        for flow in (
            "fabrication => good_market",
            "good_market => use",
            "collected => waste_market",
            "waste_market => collected",
            "fabrication => use",
            "use => eol",
            "eol => collected",
        )
    },
    **{
        flow: FlowUse(
            "market_report_alias",
            (),
            "Duplicate market-report alias, not extra global supply, use, "
            "exit or LCA demand. The MFA producer establishes alias and market closure.",
            "not_applicable",
        )
        for market in ("primary_market", "intermediate_market", "good_market", "waste_market")
        for flow in (f"sysenv => {market}", f"{market} => sysenv")
    },
}


def mfa_to_lca_flow_coverage(
    mfa: MFAEvaluation,
    *,
    foreground_scenario: str,
    background_scenario: str,
) -> pd.DataFrame:
    """Describe each admitted material flow/product and its use in the LCA.

    Rows distinguish the two source tables, aggregate polymer quantities and
    constituent source/destination axes. Counts and Mt totals cover the complete
    parent period, including history; 2019--2100 events generate reported
    LCA burdens. Nominal C and Other Elements sum to polymer mass in flows.parquet.
    Select the source table, quantity, and process step required for the analysis.

    The MFA producer owns mass balance; completed parents are trusted.
    Missing maintained flow definitions
    remain ordinary programming errors here. Related burden/provider lists are
    explanatory references; classification states whether a row drives a demand.
    """
    aggregate = mfa.model_flows.loc[:, ["year", "source_flow_name", "amount", "unit"]].assign(
        product="polymer"
    )
    constituents = mfa.material_realization.loc[
        :, ["year", "flow_name", "product", "amount", "unit"]
    ].rename(columns={"flow_name": "source_flow_name"})
    rows: list[dict[str, object]] = []
    for source_table, frame, uses in (
        ("flows.parquet", aggregate, _FLOW_USES),
        ("material_realization.parquet", constituents, _MATERIAL_USES),
    ):
        for raw_key, selected in frame.groupby(
            ["source_flow_name", "product"], observed=True, sort=True
        ):
            flow, product = cast(tuple[str, str], raw_key)
            use = uses[flow]
            specs = [_SPECS[demand] for demand in use.demands]
            rows.append(
                {
                    "foreground_scenario": foreground_scenario,
                    "background_scenario": background_scenario,
                    "source_table": source_table,
                    "source_flow_name": flow,
                    "product": product,
                    "constituent_axis": use.constituent_axis,
                    "flow_classification": use.classification,
                    "first_year": int(selected["year"].min()),
                    "last_year": int(selected["year"].max()),
                    "persisted_row_count": len(selected),
                    "nonzero_row_count": int(selected["amount"].ne(0.0).sum()),
                    "total_amount": fsum(float(value) for value in selected["amount"]),
                    "unit": str(selected["unit"].iloc[0]),
                    "burden_demand_types_json": json.dumps(use.demands, separators=(",", ":")),
                    "provider_demand_types_json": json.dumps(
                        sorted({spec.provider_demand_type or spec.demand_type for spec in specs}),
                        separators=(",", ":"),
                    ),
                    "owned_stage": "; ".join(sorted({spec.stage_group for spec in specs})),
                    "mapping_basis": "; ".join(sorted({spec.mapping_basis for spec in specs})),
                    "physical_event_basis": "; ".join(
                        sorted({spec.physical_event_basis for spec in specs})
                    ),
                    "cutoff_cycle_role": "; ".join(
                        sorted({spec.cutoff_cycle_role for spec in specs})
                    ),
                    "reason": use.reason,
                }
            )
    return pd.DataFrame(rows)
