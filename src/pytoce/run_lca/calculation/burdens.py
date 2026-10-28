"""Non-overlapping cut-off LCA burdens from the accepted MFA result."""

from dataclasses import dataclass
from math import fsum
from operator import attrgetter
from typing import cast

import numpy as np
import pandas as pd

from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _MOISTURE_REGAIN,
)
from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
    mismanaged_fate_authorities,
)
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.model.mfa import MFACandidateEvaluation, MFAEvaluation

from .custody import (
    _PA_FIBRE_GRADES,
    ULTIMATE_CARBON_ORIGINS,
    CarbonCustodyTrace,
    virgin_route_shares,
)
from .pa_receiving import PaDisposalEvent, PaRecyclingEvent, PaSpinningEvent
from .providers import MaterialCarbon

MT_TO_KG = 1_000_000_000.0
REPORTING_START_YEAR = 2019
REPORTING_END_YEAR = 2100
BURDEN_METHOD_IDENTITY = "pytoce_constituent_cutoff_burden_mapping_v4"


@dataclass(frozen=True, slots=True)
class BurdenDemands:
    """Origin-attributed provider demands and complete PA event mixtures.

    ``provider_demands`` records service reference quantities in kg and
    separately attributed carbon quantities in kg C.
    PA events preserve complete manufacturing, recycling and disposal mixtures
    for treatment at the selected background anchor. Use those complete event mixtures for receiving
    calculations.
    """

    provider_demands: pd.DataFrame
    pa_spinning_events: tuple[PaSpinningEvent, ...]
    pa_recycling_events: tuple[PaRecyclingEvent, ...]
    pa_disposal_events: tuple[PaDisposalEvent, ...]


@dataclass(frozen=True, slots=True)
class BurdenSpec:
    """Map one set of MFA flows to one foreground LCA burden.

    ``demand_type`` names the requested provider service and
    ``source_flow_names`` names the corresponding aggregate MFA movements.
    Constituent treatment and independent monomer quantities are read from
    ``material_realization``; these aggregate names are provenance labels. ``custody_status``,
    ``mapping_basis``, and ``lca_boundary`` state what mass is owned and its
    kilogram reference flow. ``burden_attribution_basis`` and ``stage_group``
    determine reporting, while ``cutoff_cycle_role`` and
    ``physical_event_basis`` preserve the material-cycle boundary and event
    year. Treatment burdens additionally use ``lca_fate``, ``fate_origin``,
    ``provider_demand_type``, and ``source_allocation_column``. The
    ``material_cycle_source`` distinguishes primary resin from mechanically or
    chemically recycled material. MFA amounts are in Mt and are converted to
    kg before provider demand is assembled.
    """

    demand_type: str
    source_flow_names: tuple[str, ...]
    custody_status: str
    mapping_basis: str
    lca_boundary: str
    burden_attribution_basis: str
    stage_group: str
    cutoff_cycle_role: str
    physical_event_basis: str
    lca_fate: str = ""
    fate_origin: str = ""
    provider_demand_type: str = ""
    source_allocation_column: str = ""
    material_cycle_source: str = "not_applicable"


BURDEN_SPECS = (
    BurdenSpec(
        "virgin_resin_production",
        ("new_polymer => processing", "new_polymer => primary_market"),
        "new_polymer_output",
        "kg_virgin_resin_output",
        "cradle_to_gate_constituent_resin_by_virgin_technology",
        "new_polymer_without_recovered_monomers",
        "material_production",
        "primary_material_cycle",
        "year new polymer is produced",
        material_cycle_source="primary_resin",
    ),
    BurdenSpec(
        "polymerization_operation",
        ("new_polymer => processing", "new_polymer => primary_market"),
        "new_polymer_output",
        "kg_destination_polymer_output",
        "native_polymerization_excluding_ethylene_and_propylene_inputs",
        "all_new_polymer_output",
        "material_production",
        "shared_polymerization",
        "year new polymer is produced",
        material_cycle_source="new_polymer",
    ),
    BurdenSpec(
        "virgin_monomer_supply",
        ("new_polymer => processing", "new_polymer => primary_market"),
        "virgin_monomer_carbon",
        "kg_destination_polymer_output",
        "native_allocated_monomer_supply_for_virgin_fraction",
        "independent_ethylene_or_propylene_supply",
        "material_production",
        "primary_material_cycle",
        "year new polymer is produced",
        material_cycle_source="virgin_monomer",
    ),
    BurdenSpec(
        "recovered_monomer_supply",
        ("new_polymer => processing", "new_polymer => primary_market"),
        "retained_recovered_monomer_carbon",
        "kg_destination_polymer_output",
        "allocated_cracker_operation_for_recovered_fraction",
        "independent_ethylene_or_propylene_supply",
        "material_production",
        "secondary_material_cycle_entry",
        "year new polymer is produced",
        material_cycle_source="recovered_monomer",
    ),
    BurdenSpec(
        "mechanical_recycling_operation",
        ("collected => reclmech",),
        "mechanical_recycling_input",
        "kg_mechanical_recycling_input",
        "cutoff_secondary_material_collection_sorting_and_recycling",
        "mechanical_recycling_input",
        "recycling_operations",
        "secondary_material_cycle_entry",
        "year waste enters mechanical recycling",
        material_cycle_source="mechanically_recycled_material",
    ),
    BurdenSpec(
        "chemical_recycling_preparation",
        ("collected => reclchem",),
        "chemical_recycling_input",
        "kg_tracked_plastic_chemical_recycling_input",
        "cutoff_secondary_material_preparation_at_treatment_facility",
        "chemical_recycling_input_operation",
        "recycling_operations",
        "secondary_material_cycle_entry",
        "year waste enters chemical recycling",
        material_cycle_source="chemically_recycled_feedstock",
    ),
    BurdenSpec(
        "chemical_recycling_conversion",
        ("collected => reclchem",),
        "chemical_recycling_input",
        "kg_tracked_plastic_chemical_recycling_input",
        "hydrothermal_inventory_per_kg_chemical_preparation_input",
        "chemical_recycling_input_operation",
        "recycling_operations",
        "secondary_material_cycle_entry",
        "year waste enters chemical recycling",
        material_cycle_source="chemically_recycled_feedstock",
    ),
    BurdenSpec(
        "incineration_treatment",
        ("collected => incineration",),
        "incineration_route",
        "kg_waste_input",
        "waste_treatment_incineration_cutoff",
        "waste_treatment_input",
        "managed_terminal_end_of_life",
        "waste_producer_terminal_treatment",
        "year collected end-of-life waste is incinerated",
        "incineration",
        "remind_mfa_physical_route",
    ),
    BurdenSpec(
        "mechanical_recycling_reject_incineration_treatment",
        ("reclmech => incineration",),
        "mechanical_recycling_managed_reject",
        "kg_waste_input",
        "waste_treatment_incineration_cutoff",
        "managed_mechanical_recycling_reject",
        "recycling_residue_treatment",
        "recycling_residue_treatment",
        "same year as the mechanical-recycling operation that generates the reject",
        "incineration",
        "remind_mfa_physical_route",
        "incineration_treatment",
    ),
    BurdenSpec(
        "landfill_treatment",
        ("collected => landfill",),
        "landfill_input",
        "kg_waste_input",
        "waste_treatment_landfill_cutoff",
        "waste_treatment_input",
        "managed_terminal_end_of_life",
        "waste_producer_terminal_treatment",
        "year collected end-of-life waste is landfilled",
        "landfill",
        "remind_mfa_physical_route",
    ),
    BurdenSpec(
        "mechanical_recycling_reject_landfill_treatment",
        ("reclmech => landfill",),
        "mechanical_recycling_landfilled_reject",
        "kg_waste_input",
        "waste_treatment_landfill_cutoff",
        "managed_mechanical_recycling_reject",
        "recycling_residue_treatment",
        "recycling_residue_treatment",
        "same year as the mechanical-recycling operation that generates the remainder",
        "landfill",
        "constituent_recovery_managed_fate",
        "landfill_treatment",
    ),
    BurdenSpec(
        "processing_conversion",
        (
            "processing => fabrication",
            "processing => intermediate_market",
        ),
        "material_carbon",
        "kg_processed_plastic",
        "processing_conversion",
        "conversion_service_output",
        "processing_conversion",
        "material_conversion",
        "year processed plastic leaves the processing node",
    ),
    BurdenSpec(
        "mismanaged_open_burning_treatment",
        ("eol => mismanaged",),
        "mismanaged_end_of_life_input",
        "kg_waste_input",
        "direct_uncontrolled_open_burning_cutoff",
        "actual_mismanaged_waste_input",
        "mismanaged_end_of_life",
        "waste_producer_uncontrolled_treatment",
        "year end-of-life waste becomes mismanaged",
        "open_burning",
        "cottom_2024_global_uncollected_waste_split",
        "mismanaged_open_burning_treatment",
        "open_burning_share",
    ),
    BurdenSpec(
        "mismanaged_open_dump_treatment",
        ("eol => mismanaged",),
        "mismanaged_end_of_life_input",
        "kg_waste_input",
        "direct_uncontrolled_open_dump_cutoff",
        "actual_mismanaged_waste_input",
        "mismanaged_end_of_life",
        "waste_producer_uncontrolled_treatment",
        "year end-of-life waste becomes mismanaged",
        "open_dump",
        "cottom_2024_nonburned_residual_project_lci_proxy",
        "mismanaged_open_dump_treatment",
        "open_dump_share",
    ),
)


_SPEC_BY_DEMAND = {spec.demand_type: spec for spec in BURDEN_SPECS}


def pa_event_burden(event: PaSpinningEvent | PaRecyclingEvent | PaDisposalEvent) -> BurdenSpec:
    """Use the PA event's existing stage, cycle and fate definitions.

    Spinning residues stay with conversion, and washing residues with mechanical
    recycling. A separately disposed end-of-life or dry reject mixture retains
    its actual treatment stage.
    """
    if isinstance(event, PaSpinningEvent):
        demand = "processing_conversion"
    elif isinstance(event, PaRecyclingEvent):
        demand = "mechanical_recycling_operation"
    else:
        demand = event.demand_type
    return _SPEC_BY_DEMAND[demand]


_GROUP_COLUMNS = [
    "foreground_scenario",
    "year",
    "region",
    "material",
    "good",
    "custody",
]

_MECHANICAL_RECYCLING_FLOWS = {
    "collected => reclmech": "mechanical_recycling_input_kg",
    "reclmech => processing": "mechanical_recycling_output_kg",
    "reclmech => incineration": "mechanical_recycling_managed_reject_incinerated_kg",
    "reclmech => landfill": "mechanical_recycling_managed_reject_landfilled_kg",
    "reclmech => waterborne_loss": "mechanical_recycling_waterborne_loss_kg",
    "reclmech => airborne_loss": "mechanical_recycling_airborne_loss_kg",
}


def _with_mechanical_recycling_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Sum the four distinct principal losses and calculate shares of dry input."""
    result = frame.copy()
    result["mechanical_recycling_total_loss_kg"] = (
        result["mechanical_recycling_managed_reject_incinerated_kg"]
        + result["mechanical_recycling_managed_reject_landfilled_kg"]
        + result["mechanical_recycling_waterborne_loss_kg"]
        + result["mechanical_recycling_airborne_loss_kg"]
    )
    result["mechanical_recycling_output_share_of_input"] = result[
        "mechanical_recycling_output_kg"
    ].div(result["mechanical_recycling_input_kg"])
    result["mechanical_recycling_total_loss_share_of_input"] = result[
        "mechanical_recycling_total_loss_kg"
    ].div(result["mechanical_recycling_input_kg"])
    return result


def mechanical_recycling_loss_result(accepted_flows: pd.DataFrame) -> pd.DataFrame:
    """Return annual and cumulative principal recovery and loss relationships.

    Dry principal amounts come directly from the completed MFA parent. Managed
    incineration and landfill remain separate from wastewater-inlet and airborne
    losses. Wastewater-inlet quantities refer to the treatment inlet;
    LCA reports the emissions of the receiving treatment.
    """

    selected = accepted_flows.loc[
        accepted_flows["year"].between(REPORTING_START_YEAR, REPORTING_END_YEAR)
        & accepted_flows["element"].isin(("C", "Other Elements"))
        & accepted_flows["source_flow_name"].isin(_MECHANICAL_RECYCLING_FLOWS)
    ]
    coordinate_columns = list(_GROUP_COLUMNS)
    annual = (
        selected.groupby(
            [*coordinate_columns, "source_flow_name"],
            observed=True,
            sort=True,
        )["amount"]
        .sum()
        .unstack("source_flow_name", fill_value=cast(int, 0.0))
        .reindex(columns=list(_MECHANICAL_RECYCLING_FLOWS), fill_value=0.0)
        .rename(columns=_MECHANICAL_RECYCLING_FLOWS)
        .mul(MT_TO_KG)
        .reset_index()
    )
    annual = annual.loc[annual["mechanical_recycling_input_kg"].gt(0.0)]
    amount_columns = list(_MECHANICAL_RECYCLING_FLOWS.values())
    annual = _with_mechanical_recycling_shares(annual)
    annual.insert(1, "reporting_period", "annual")

    cumulative_keys = [column for column in _GROUP_COLUMNS if column != "year"]
    cumulative = (
        annual.groupby(cumulative_keys, observed=True, sort=True)[amount_columns]
        .sum()
        .reset_index()
    )
    cumulative = _with_mechanical_recycling_shares(cumulative)
    cumulative.insert(1, "reporting_period", "cumulative_2019_2100")
    cumulative["year"] = pd.NA

    result = pd.concat((annual, cumulative), ignore_index=True)
    result["year"] = result["year"].astype("Int64")
    result["unit"] = "kilogram"
    return result.sort_values(
        ["reporting_period", "year", "region", "material", "good"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


# These coordinates belong to the admitted material-realization table. Sparse
# indexes below omit numerical zeros.
_TREATMENT_QUANTITIES = {
    "mechanical_recycling_operation": ("mechanical_treatment", "collected => reclmech"),
    "chemical_recycling_preparation": ("chemical_treatment", "collected => reclchem"),
    "chemical_recycling_conversion": ("chemical_treatment", "reclchem => chemical_feedstock"),
    "incineration_treatment": ("incineration_treatment", "collected => incineration"),
    "landfill_treatment": ("landfill_treatment", "collected => landfill"),
    "mechanical_recycling_reject_incineration_treatment": (
        "mechanical_incinerated_reject",
        "reclmech => incineration",
    ),
    "mechanical_recycling_reject_landfill_treatment": (
        "mechanical_landfilled_reject",
        "reclmech => landfill",
    ),
}


def _origin_demands(
    spec: BurdenSpec,
    *,
    scenario: str,
    year: int,
    region: str,
    material: str,
    constituent: str,
    product: str,
    parent_amount_mt: float,
    allocation: float,
    carbon_mt: np.ndarray,
    virgin_route: str = "all",
) -> list[dict[str, object]]:
    """Attribute one service demand over its already allocated carbon.

    ``parent_amount_mt * allocation`` is the service-reference quantity before
    origin attribution. ``carbon_mt`` already includes this allocation and, for
    monomer supply, retention. The origin fraction partitions an otherwise
    identical operation quantity. Provider routes use it to select the declared
    carbon-flow identities for treated plastic; unrelated process burdens retain
    their native identities. Conversion performance is assumed independent of
    carbon origin within each constituent and operation.
    """
    amount = parent_amount_mt * allocation
    if amount == 0:
        return []
    total_carbon = fsum(carbon_mt)
    rows = []
    for origin, carbon in zip(ULTIMATE_CARBON_ORIGINS, carbon_mt, strict=True):
        fraction = float(carbon) / total_carbon
        if fraction == 0:
            continue
        rows.append(
            {
                "foreground_scenario": scenario,
                "year": year,
                "region": region,
                "material": material,
                "constituent": constituent,
                "product": product,
                "good": "all",
                "custody": "h12_physical_custody",
                "demand_type": spec.demand_type,
                "provider_demand_type": spec.provider_demand_type or spec.demand_type,
                "basis_flow_names": " + ".join(spec.source_flow_names),
                "custody_status": spec.custody_status,
                "source_parent_amount_kg": parent_amount_mt * MT_TO_KG,
                "source_allocation_share": allocation,
                "amount_kg": amount * fraction * MT_TO_KG,
                "mfa_carbon_amount_kg": float(carbon) * MT_TO_KG,
                "virgin_feedstock_route": virgin_route,
                "ultimate_carbon_origin": origin,
                "origin_fraction": fraction,
                "mapping_basis": spec.mapping_basis,
                "lca_boundary": spec.lca_boundary,
                "burden_attribution_basis": spec.burden_attribution_basis,
                "stage_group": spec.stage_group,
                "cutoff_cycle_role": spec.cutoff_cycle_role,
                "physical_event_basis": spec.physical_event_basis,
                "physical_event_year": year,
                "material_cycle_source": spec.material_cycle_source,
                "lca_fate": spec.lca_fate,
                "fate_origin": spec.fate_origin,
                "allocation_system_model": "ecoinvent_3_12_allocation_cutoff_by_classification",
            }
        )
    return rows


def build_burden_demands(
    mfa: MFAEvaluation | MFACandidateEvaluation,
    material_carbon: MaterialCarbon,
    trace: CarbonCustodyTrace,
) -> BurdenDemands:
    """Map admitted constituents and carbon to their direct LCI owners.

    All material quantities are Mt; the returned long rows contain kg service
    reference quantity and kg C. Each eligible polymer has one native operation
    and independent ethylene/propylene supply pairs. Virgin technology and
    ultimate carbon origin are separate axes. Input treatment uses actual
    constituent amounts after selective recycling; family shares apply only
    to family-level mismanaged and finished-manufacturing quantities.
    """
    composition = load_material_composition().constituents
    shares = virgin_route_shares(mfa.nominal_feedstock_equivalents)
    carbon_values = np.asarray(material_carbon.fractions)
    tps_formulations: dict[tuple[int, str, str, str, str], list[tuple[float, float, float]]] = {}
    for (
        year,
        region,
        material,
        constituent,
        _good,
        flow,
    ), amounts in trace.tps_route_amounts.items():
        if flow in {
            "collected => reclmech",
            "reclmech => landfill",
            "collected => landfill",
            "collected => incineration",
            "eol => mismanaged",
        }:
            tps_formulations.setdefault((year, region, material, constituent, flow), []).append(
                amounts
            )
    carbon_index = (
        trace.calculation.groupby(
            [
                "year",
                "region",
                "material",
                "constituent",
                "product",
                "source_flow_name",
                "ultimate_carbon_origin",
            ],
            observed=True,
            sort=False,
        )["carbon_amount_mt"]
        .agg(fsum)
        .to_dict()
    )
    physical = mfa.model_flows.loc[
        mfa.model_flows["year"].between(trace.first_year, REPORTING_END_YEAR)
    ]
    pooled = (
        physical.groupby(
            ["year", "region", "material", "source_flow_name"],
            observed=True,
            sort=False,
        )["amount"]
        .agg(fsum)
        .to_dict()
    )
    scenario = str(mfa.model_flows["foreground_scenario"].iloc[0])
    fate = mismanaged_fate_authorities()
    fate_shares = fate.set_index("region").to_dict("index")
    table = mfa.material_realization.loc[
        mfa.material_realization["year"].between(trace.first_year, REPORTING_END_YEAR)
        & mfa.material_realization["amount"].ne(0)
    ].copy()
    table["material"] = table["source_material"].fillna(table["destination_material"])
    table["constituent"] = table["source_constituent"].fillna(table["destination_constituent"])
    quantities = table.set_index(
        ["year", "region", "material", "constituent", "product", "flow_name"]
    )["amount"].to_dict()
    events = physical[["year", "region"]].drop_duplicates().itertuples(index=False, name=None)
    rows: list[dict[str, object]] = []
    pa_spinning_events: list[PaSpinningEvent] = []
    pa_recycling_events: list[PaRecyclingEvent] = []
    pa_disposal_events: list[PaDisposalEvent] = []
    for year, region in events:
        for index, item in enumerate(composition):
            coordinate = (int(year), str(region), item.material, item.constituent)
            polymer_coordinate = (*coordinate, "polymer")
            pa_grade = _PA_FIBRE_GRADES.get((item.material, item.constituent))
            dose_coordinate = (*coordinate, "all")

            def carbon_for(*flows: str, product: str = "polymer") -> np.ndarray:
                """Sum actual carbon amounts, preserving constituent and product axes."""
                return np.array(
                    [
                        fsum(
                            float(carbon_index.get((*coordinate, product, flow, origin), 0.0))
                            for flow in flows
                        )
                        for origin in ULTIMATE_CARBON_ORIGINS
                    ]
                )

            def pa_exit_mass(flow: str) -> tuple[float, float, float]:
                """Convert an actual PA principal exit from Mt C to kg material.

                Both PA6 C6H11NO and PA66 C12H22N2O2 retain the admitted
                nominal carbon fraction 72/113. Use the dedicated exit
                carbon, including imports.
                """
                return cast(
                    tuple[float, float, float],
                    tuple(float(value) * MT_TO_KG / (72 / 113) for value in carbon_for(flow)),
                )

            def emit(
                demand: str,
                amount: float,
                carbon: np.ndarray,
                *,
                allocation: float = 1.0,
                product: str = "polymer",
                virgin_route: str = "all",
            ) -> None:
                """Append one constituent event."""
                # PA receiving consumes the complete actual mixture instead of a
                # fixed treatment service. Constituent custody uses 72/113 kg C/kg
                # PA principal; the finish dose in kg is a separate quantity.
                if pa_grade is not None and demand == "mechanical_recycling_operation":
                    pa_recycling_events.append(
                        PaRecyclingEvent(
                            year=int(year),
                            region=str(region),
                            constituent=item.constituent,
                            principal_by_origin_kg=pa_exit_mass("collected => reclmech"),
                            embedded_finish_dose_kg=trace.pa_finish_amounts.get(
                                (*dose_coordinate, "collected => reclmech"), 0.0
                            )
                            * MT_TO_KG,
                            recovered_principal_kg=float(
                                quantities.get((*polymer_coordinate, "mechanical_resin"), 0.0)
                            )
                            * MT_TO_KG,
                            airborne_principal_by_origin_kg=pa_exit_mass(
                                "reclmech => airborne_loss"
                            ),
                            airborne_finish_dose_kg=trace.pa_finish_amounts.get(
                                (*dose_coordinate, "reclmech => airborne_loss"), 0.0
                            )
                            * MT_TO_KG,
                        )
                    )
                    return
                if pa_grade is not None and demand in {
                    "incineration_treatment",
                    "landfill_treatment",
                    "mechanical_recycling_reject_incineration_treatment",
                    "mismanaged_open_burning_treatment",
                    "mismanaged_open_dump_treatment",
                }:
                    managed = demand in _TREATMENT_QUANTITIES
                    carbon_flow = (
                        _TREATMENT_QUANTITIES[demand][1] if managed else "eol => mismanaged"
                    )
                    principal = cast(
                        tuple[float, float, float],
                        tuple(float(value) * MT_TO_KG / (72 / 113) for value in carbon),
                    )
                    # Ordinary EOL fiber carries one surface dose and moisture
                    # per principal kg. Recycling rejects are dry; their outgoing
                    # embedded dose already includes retained surface after washing.
                    # The fixed pa_granulate recovery recipe sends all managed
                    # rejects to incineration; only TPS has landfilled rejects.
                    dry_reject = demand == "mechanical_recycling_reject_incineration_treatment"
                    surface = 0.0 if dry_reject else fsum(principal)
                    dose = trace.pa_finish_amounts.get((*dose_coordinate, carbon_flow), 0.0)
                    pa_disposal_events.append(
                        PaDisposalEvent(
                            year=int(year),
                            region=str(region),
                            constituent=item.constituent,
                            demand_type=demand,
                            principal_by_origin_kg=principal,
                            total_finish_dose_kg=surface + dose * MT_TO_KG * allocation,
                            water_kg=_MOISTURE_REGAIN[pa_grade] * surface,
                        )
                    )
                    return
                rows.extend(
                    _origin_demands(
                        _SPEC_BY_DEMAND[demand],
                        scenario=scenario,
                        year=int(year),
                        region=str(region),
                        material=item.material,
                        constituent=item.constituent,
                        product=product,
                        parent_amount_mt=amount,
                        allocation=allocation,
                        carbon_mt=carbon,
                        virgin_route=virgin_route,
                    )
                )

            output = fsum(
                float(quantities.get((*polymer_coordinate, flow), 0.0))
                for flow in ("new_polymer_domestic", "new_polymer_export")
            )
            route_shares = shares[(int(year), str(region), item.material)]
            if output > 0:
                if item.chemical_polymer is None:
                    for r, virgin_route in enumerate(ULTIMATE_CARBON_ORIGINS):
                        emit(
                            "virgin_resin_production",
                            output,
                            output * route_shares[r] * carbon_values[index, r],
                            allocation=float(route_shares[r]),
                            virgin_route=virgin_route,
                        )
                else:
                    emit(
                        "polymerization_operation",
                        output,
                        carbon_for("new_polymer => processing", "new_polymer => primary_market"),
                    )
                    for monomer in ("ethylene", "propylene"):
                        monomer_coordinate = (*coordinate, monomer)
                        virgin = float(
                            quantities.get((*monomer_coordinate, "virgin_monomer_consumption"), 0.0)
                        )
                        recovered = float(
                            quantities.get(
                                (*monomer_coordinate, "recovered_monomer_consumption"), 0.0
                            )
                        )
                        gross = virgin + recovered
                        # Positive new polymer has positive admitted E and P requirements.
                        for r, virgin_route in enumerate(ULTIMATE_CARBON_ORIGINS):
                            carbon = np.zeros(3)
                            carbon[r] = (
                                trace.chemistry.retention[index]
                                * trace.chemistry.monomers[monomer]
                                * virgin
                                * route_shares[r]
                            )
                            emit(
                                "virgin_monomer_supply",
                                output,
                                carbon,
                                allocation=virgin / gross * float(route_shares[r]),
                                product=monomer,
                                virgin_route=virgin_route,
                            )
                        emit(
                            "recovered_monomer_supply",
                            output,
                            carbon_for("recovered_monomer_consumption", product=monomer),
                            allocation=recovered / gross,
                            product=monomer,
                        )
            for demand, (quantity, carbon_flow) in _TREATMENT_QUANTITIES.items():
                amount = float(quantities.get((*polymer_coordinate, quantity), 0.0))
                if amount > 0:
                    if (
                        demand
                        in {
                            "mechanical_recycling_operation",
                            "mechanical_recycling_reject_landfill_treatment",
                            "landfill_treatment",
                            "incineration_treatment",
                        }
                        and item.mechanical_recovery == "tps_compound"
                    ):
                        # TPS carries its original starch/glycerol formulation
                        # through trade, use and recovery. Biogenic primer in a
                        # synthetic formulation cannot select native-starch waste.
                        # Stored principal mass already includes the actual reject
                        # or disposal fraction. The provider owns whole wet mass;
                        # this principal demand uses that prepared quantity.
                        for r, route in enumerate(ULTIMATE_CARBON_ORIGINS):
                            route_amount = fsum(
                                part[r] for part in tps_formulations[(*coordinate, carbon_flow)]
                            )
                            emit(
                                demand,
                                route_amount,
                                route_amount * carbon_values[index, r],
                                virgin_route=route,
                            )
                    else:
                        emit(demand, amount, carbon_for(carbon_flow))
            mismanaged = float(
                pooled[(int(year), str(region), item.material, "eol => mismanaged")]
            ) * float(item.share)
            if mismanaged > 0:
                carbon = carbon_for("eol => mismanaged")
                for demand in (
                    "mismanaged_open_burning_treatment",
                    "mismanaged_open_dump_treatment",
                ):
                    fraction = float(
                        fate_shares[str(region)][_SPEC_BY_DEMAND[demand].source_allocation_column]
                    )
                    if item.mechanical_recovery == "tps_compound":
                        # Both TPS fates partition the same historical principal
                        # mass. Both providers receive the complete formulation;
                        # dumping retains qualified native emissions/degradation.
                        # Keep the formulation on both parent records, including
                        # its biogenic primer. Apply each fate share once;
                        # constituent shares are already present in traced mass.
                        for r, route in enumerate(ULTIMATE_CARBON_ORIGINS):
                            route_amount = fsum(
                                part[r]
                                for part in tps_formulations[(*coordinate, "eol => mismanaged")]
                            )
                            emit(
                                demand,
                                route_amount,
                                route_amount * carbon_values[index, r] * fraction,
                                allocation=fraction,
                                virgin_route=route,
                            )
                    else:
                        emit(demand, mismanaged, carbon * fraction, allocation=fraction)

            conversion_flows = _SPEC_BY_DEMAND["processing_conversion"].source_flow_names
            finished = float(item.share) * fsum(
                float(pooled[(int(year), str(region), item.material, flow)])
                for flow in conversion_flows
            )
            if finished > 0:
                if pa_grade is not None:
                    pa_spinning_events.append(
                        PaSpinningEvent(
                            year=int(year),
                            region=str(region),
                            constituent=item.constituent,
                            finished_principal_kg=float(
                                quantities[(*polymer_coordinate, "manufacturing_finished_output")]
                            )
                            * MT_TO_KG,
                            gross_principal_kg=float(
                                quantities[(*polymer_coordinate, "manufacturing_principal_input")]
                            )
                            * MT_TO_KG,
                            recovered_principal_kg=float(
                                quantities.get((*polymer_coordinate, "mechanical_resin"), 0.0)
                            )
                            * MT_TO_KG,
                            # The sparse custody graph omits a zero recovery
                            # edge: no recovered polymer means no embedded dose.
                            embedded_finish_dose_kg=trace.pa_finish_amounts.get(
                                (*dose_coordinate, "reclmech => processing"), 0.0
                            )
                            * MT_TO_KG,
                            maintenance_principal_by_origin_kg=pa_exit_mass(
                                "processing => spinning_equipment_deposits"
                            ),
                            maintenance_finish_dose_kg=trace.pa_finish_amounts[
                                (*dose_coordinate, "processing => spinning_equipment_deposits")
                            ]
                            * MT_TO_KG,
                            finished_principal_carbon_by_origin_kg=cast(
                                tuple[float, float, float],
                                tuple(
                                    float(value) * MT_TO_KG
                                    for value in carbon_for("processing => finished_principal")
                                ),
                            ),
                            caprolactam_air_by_origin_kg=pa_exit_mass(
                                "processing => caprolactam_air"
                            ),
                            caprolactam_water_by_origin_kg=pa_exit_mass(
                                "processing => caprolactam_collected_water"
                            ),
                        )
                    )
                emit("processing_conversion", finished, carbon_for(*conversion_flows))
    columns = [
        "foreground_scenario",
        "year",
        "region",
        "material",
        "constituent",
        "product",
        "good",
        "custody",
        "demand_type",
        "provider_demand_type",
        "basis_flow_names",
        "custody_status",
        "source_parent_amount_kg",
        "source_allocation_share",
        "amount_kg",
        "mfa_carbon_amount_kg",
        "virgin_feedstock_route",
        "ultimate_carbon_origin",
        "origin_fraction",
        "mapping_basis",
        "lca_boundary",
        "burden_attribution_basis",
        "stage_group",
        "cutoff_cycle_role",
        "physical_event_basis",
        "physical_event_year",
        "material_cycle_source",
        "lca_fate",
        "fate_origin",
        "allocation_system_model",
    ]
    provider_demands = pd.DataFrame(rows, columns=columns).sort_values(
        [
            "year",
            "region",
            "material",
            "constituent",
            "demand_type",
            "product",
            "virgin_feedstock_route",
            "ultimate_carbon_origin",
        ],
        kind="stable",
        ignore_index=True,
    )
    return BurdenDemands(
        provider_demands,
        tuple(sorted(pa_spinning_events, key=attrgetter("year", "region", "constituent"))),
        tuple(sorted(pa_recycling_events, key=attrgetter("year", "region", "constituent"))),
        tuple(
            sorted(
                pa_disposal_events, key=attrgetter("year", "region", "constituent", "demand_type")
            )
        ),
    )
