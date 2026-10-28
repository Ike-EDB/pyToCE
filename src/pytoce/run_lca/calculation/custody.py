"""Track constituent carbon through polymer flows, monomers and historical cohorts.

Polymer transfers carry the source carbon density. Separate recovered ethylene and propylene pools
retain
source origins; the adopted equal effective monomer retention determines carbon
incorporated in new polyolefins.
Thermoplastic starch (TPS) retains its three source-route principal masses in
the same mixing equations. Reporting converts them forward to carbon origins.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import fsum
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, eye  # type: ignore[import-untyped]
from scipy.sparse.linalg import splu  # type: ignore[import-untyped]

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _WASH_REMAINING,
)
from pytoce.run_mfa.model.materials import MaterialConstituent, load_material_composition
from pytoce.run_mfa.model.mfa import MFACandidateEvaluation, MFAEvaluation

from .providers import MaterialCarbon

ULTIMATE_CARBON_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
CARBON_CUSTODY_METHOD_IDENTITY = "pytoce_manufactured_constituent_carbon_custody_v6"
Node = tuple[str, str, str, str, str]
EntryHistory = dict[tuple[str, str, str, str], dict[int, np.ndarray]]
HistoryEntry = tuple[int, str, str, str, str, float, float, float]
TPSRouteAmounts = dict[tuple[int, str, str, str, str, str], tuple[float, float, float]]
PAFinishHistoryEntry = tuple[int, str, str, str, str, float]
PAFinishEntryHistory = dict[tuple[str, str, str, str], dict[int, float]]
PAFinishAmounts = dict[tuple[int, str, str, str, str, str], float]
# These two fiber constituents receive the reconstructed surface finish.
# Uncoated Others PA and bio-PA retain their separate declared inventories.
_PA_FIBRE_GRADES = {
    ("Fibres", "polyamide_6_resin"): "pa6",
    ("Fibres", "polyamide_66_resin"): "pa66",
}
_REPORTING_START_YEAR = 2019
_REPORTING_END_YEAR = 2100
_MONOMERS = ("ethylene", "propylene")
_CUSTODY_COLUMNS = [
    "foreground_scenario",
    "year",
    "region",
    "material",
    "constituent",
    "good",
    "source_flow_name",
    "product",
    "custody_status",
    "ultimate_carbon_origin",
    "carbon_amount_mt",
    "source_carbon_amount_mt",
    "origin_fraction",
    "unit",
]
_REPORTED_FLOWS = {
    "new_polymer => processing": "new_polymer_output",
    "new_polymer => primary_market": "new_polymer_output",
    "collected => reclmech": "mechanical_recycling_input",
    "collected => reclchem": "chemical_recycling_input",
    "reclmech => processing": "mechanically_recycled_output",
    "processing => fabrication": "material_carbon",
    "processing => intermediate_market": "material_carbon",
    "fabrication => use": "entry_into_use",
    "good_market => use": "entry_into_use",
    "collected => incineration": "incineration_route",
    "reclmech => incineration": "incineration_route",
    "reclmech => waterborne_loss": "mechanical_recycling_waterborne_loss",
    "reclmech => airborne_loss": "mechanical_recycling_airborne_loss",
    "reclmech => landfill": "mechanical_recycling_landfilled_reject",
    "processing => finished_principal": "manufactured_principal_output",
    **{
        f"processing => {name}": "manufacturing_principal_loss"
        for name in (
            "styrene_air",
            "butadiene_air",
            "extrusion_dust",
            "post_release_solid",
            "container_residue",
            "methyl_methacrylate_air",
            "preparation_residue",
            "settled_overspray",
            "airborne_overspray",
            "caprolactam_air",
            "caprolactam_collected_water",
            "spinning_equipment_deposits",
            "principal_remainder",
        )
    },
    "collected => landfill": "landfill_input",
    "eol => mismanaged": "mismanaged_end_of_life_input",
    "reclchem => chemical_feedstock": "chemical_processed_feed",
    "reclchem => chemical_preparation_reject": "chemical_preparation_reject",
    "recovered_monomer_consumption": "retained_recovered_monomer_carbon",
}


@dataclass(frozen=True, slots=True)
class CarbonCustodyHistory:
    """End-2019 constituent densities by cohort and product use.

    ``entries`` contains kg C/kg constituent by ultimate carbon origin for
    non-TPS materials. ``tps_route_entries`` instead contains kg principal/kg
    TPS principal from fossil synthesis, native starch and DAC synthesis. The latter retains each
    route's complete formulation quantities, including biogenic primer carbon. ``pa_finish_entries``
    separately carries embedded
    finish dose per principal kg for coated PA fibers. New surface finish and principal carbon are
    tracked separately.
    """

    through_year: int
    entries: tuple[HistoryEntry, ...]
    tps_route_entries: tuple[HistoryEntry, ...]
    pa_finish_entries: tuple[PAFinishHistoryEntry, ...]


@dataclass(frozen=True, slots=True)
class _ChemicalCarbon:
    """Compatible monomer carbon, constituent backbone carbon and retention fractions."""

    monomers: dict[str, float]
    backbone: np.ndarray
    retention: np.ndarray


@dataclass(frozen=True, slots=True)
class _PrincipalCarbonExit:
    """Kg and incoming-principal-equivalent kg per finished principal kg."""

    name: str
    physical: float
    carbon_equivalent: float


@dataclass(frozen=True, slots=True)
class _ManufacturingCarbon:
    """Retained and lost incoming-principal equivalents per finished principal kg."""

    finished_equivalent: float
    exits: tuple[_PrincipalCarbonExit, ...]


@dataclass(frozen=True, slots=True)
class CarbonCustodyTrace:
    """Annual carbon, derived chemistry and the reusable end-2019 cohort state.

    ``first_year`` is the first reported year calculated, including when all its
    amounts are zero. Consumers reuse the derived chemistry.
    ``tps_route_amounts`` retains principal Mt by fossil synthesis, native
    starch and DAC synthesis for formulation-specific inventory quantities;
    keys are year, region, material, constituent, good and source flow.
    ``pa_finish_amounts`` uses those coordinates for Mt of embedded dose, which
    multiplies the fixed retained-finish vector. The dose is expressed relative to the original
    finish formulation.
    """

    calculation: pd.DataFrame
    history: CarbonCustodyHistory
    chemistry: _ChemicalCarbon
    first_year: int
    tps_route_amounts: TPSRouteAmounts
    pa_finish_amounts: PAFinishAmounts


@dataclass(frozen=True, slots=True)
class _CohortDiscardWeights:
    """Fractions assigning one year's discarded family mass to entry cohorts."""

    prior_years: np.ndarray
    prior_weights: np.ndarray
    current_weight: float


CohortWeightsByYear = dict[int, dict[tuple[str, str, str], _CohortDiscardWeights]]


@dataclass(frozen=True, slots=True)
class _FlowMetadata:
    """Destination or source constituent coordinates of one reported transfer."""

    region: str
    material: str
    constituent: str
    good: str
    flow: str
    product: str = "polymer"


@dataclass(frozen=True, slots=True)
class _CarbonTransfer:
    """Source-equivalent mass numerator for one carbon transfer coefficient.

    Ordinary transfers use Mt polymer. Chemical and manufacturing transfers
    include their source/product carbon ratio or retention factor; dividing by
    source throughput gives the coefficient in the carbon mixing system.
    """

    source: Node
    target: Node
    numerator: float
    metadata: _FlowMetadata | None


class _YearGraph:
    """One annual mixing system with separately accounted throughputs.

    The three columns track ultimate carbon origins except in disconnected TPS
    constituent systems, where they track the three source-route principal masses.
    """

    def __init__(self) -> None:
        """Start empty input sums, source injections and transfer terms."""
        self.mass_inputs: dict[Node, list[float]] = defaultdict(list)
        self.external: dict[Node, np.ndarray] = {}
        self.edges: list[_CarbonTransfer] = []

    def inject(self, node: Node, mass: float, origin_amounts: np.ndarray) -> None:
        """Register mass and three origin amounts in the constituent's fixed basis."""
        if mass > 0:
            self.mass_inputs[node].append(mass)
            self.external[node] = origin_amounts

    def transfer(
        self,
        source: Node,
        target: Node,
        amount: float,
        metadata: _FlowMetadata | None = None,
        *,
        physical_amount: float | None = None,
    ) -> None:
        """Carry source-equivalent Mt for carbon and actual Mt into the target.

        Ordinary transfers have equal actual material quantities and source-equivalent quantities.
        Manufacturing supplies the finished mass; chemical carbon
        edges use zero because their supply is registered separately.
        """
        if amount > 0:
            self.edges.append(_CarbonTransfer(source, target, amount, metadata))
            target_mass = amount if physical_amount is None else physical_amount
            if target_mass > 0:
                self.mass_inputs[target].append(target_mass)


def _solve_origin_mixing_network(
    graph: _YearGraph,
    *,
    year: int,
    tps_constituents: frozenset[tuple[str, str]],
) -> tuple[dict[Node, np.ndarray], np.ndarray, dict[Node, float]]:
    """Solve ``(I-T)x=e`` for three origin amounts with one shared factorization.

    Each ordinary coefficient is transferred polymer mass divided by source input
    mass. Source-only nodes have known carbon inputs; only nodes with incoming
    transfers require the shared three-origin factorization. Discard equations
    are substituted into the core and reconstructed from their positive terms,
    preserving recycling feedback and very small current-cohort origins. Empty activity years
    yield empty results. Singular systems retain the library's exception;
    nonfinite or negative amounts fail before downstream use. TPS route masses
    are normalized to their throughput, preserving exactly absent
    routes. Carbon densities of all other constituents remain unnormalized.
    """
    mass = {node: fsum(parts) for node, parts in graph.mass_inputs.items()}
    if not mass:
        return {}, np.empty((0, 3)), mass
    nodes = sorted(mass)
    positions = {node: i for i, node in enumerate(nodes)}
    external = np.zeros((len(nodes), 3))
    for node, value in graph.external.items():
        external[positions[node]] = value
    rows, columns, values = [], [], []
    for edge in graph.edges:
        rows.append(positions[edge.target])
        columns.append(positions[edge.source])
        values.append(edge.numerator / mass[edge.source])
    transfer = cast(
        Any, coo_matrix((values, (rows, columns)), shape=(len(nodes), len(nodes)))
    ).tocsc()
    known = np.diff(transfer.tocsr().indptr) == 0
    mixed = ~known
    origin_input = external.copy()
    # Source-only equations are x=e, including exact absent origins in earlier
    # cohorts. Carry these inputs into the mixing RHS before factorization;
    # eliminating them avoids roundoff in quantities already known exactly.
    right_hand_side = external[mixed] + transfer[mixed, :][:, known] @ external[known]
    mixed_indices = np.flatnonzero(mixed)
    discard = np.array([nodes[i][0] == "discard" for i in mixed_indices], dtype=bool)
    core = ~discard
    mixed_transfer = transfer[mixed, :][:, mixed]
    to_discard = mixed_transfer[discard, :][:, core]
    from_discard = mixed_transfer[core, :][:, discard]
    # Discard receives historical origins plus use-to-discard transfers. Substitute x_D=b_D+T_DR*x_R
    # into the core equations: (I-T_RR-T_RD*T_DR)*x_R=b_R+T_RD*b_D. This retains recycling feedback
    # while reconstructing tiny current-cohort amounts through the reduced linear system.
    system = (
        eye(int(np.count_nonzero(core)), format="csc")
        - mixed_transfer[core, :][:, core]
        - from_discard @ to_discard
    ).tocsc()
    core_origins = splu(system).solve(
        right_hand_side[core] + from_discard @ right_hand_side[discard]
    )
    origin_input[mixed_indices[core]] = core_origins
    origin_input[mixed_indices[discard]] = right_hand_side[discard] + to_discard @ core_origins
    if not np.isfinite(origin_input).all() or np.any(origin_input < 0):
        raise ArithmeticError(f"Constituent mixing returned invalid origin amounts in {year}.")
    # TPS transfers are homogeneous and have no recovered-monomer edges. Its
    # three nonnegative source-route amounts partition the input mass;
    # normalization removes solver roundoff.
    for node, amounts in zip(nodes, origin_input, strict=True):
        if (node[2], node[3]) in tps_constituents:
            amounts *= mass[node] / fsum(amounts)
    # Apply the transfer fraction first: multiplying two Mt
    # quantities can overflow even when the transferred carbon in Mt is finite.
    edge_amounts = np.asarray(
        [
            origin_input[positions[edge.source]] * (edge.numerator / mass[edge.source])
            for edge in graph.edges
        ]
    ).reshape(-1, 3)
    return dict(zip(nodes, origin_input, strict=True)), edge_amounts, mass


def _solve_pa_finish_custody(
    graph: _YearGraph,
    mass: dict[Node, float],
    year: int,
    cohort_weights: dict[tuple[str, str, str], _CohortDiscardWeights],
    history: PAFinishEntryHistory,
    deposit_fractions: dict[tuple[str, str], float],
) -> tuple[dict[Node, float], PAFinishAmounts]:
    """Carry one embedded finish dose through actual PA markets and cohorts.

    The fixed retained compound vector is supplied at the receiving boundary.
    Newly applied surface material stays outside this scalar until recycling:
    the unwashed fraction then enters the melt with inherited embedded finish.
    Selective caprolactam exits contain no finish; equipment deposits
    contain the same embedded fraction as their principal feed. Principal and
    auxiliary equations therefore share routes but not every transfer.
    """
    nodes = sorted(node for node in mass if (node[2], node[3]) in _PA_FIBRE_GRADES)
    if not nodes:
        return {}, {}
    positions = {node: i for i, node in enumerate(nodes)}
    external = np.zeros(len(nodes))
    for node, position in positions.items():
        stage, region, material, constituent, good = node
        if stage == "reclmech":
            # Washing acts only on the one surface dose per incoming principal
            # kg. The retained fraction becomes embedded in the recovered melt;
            # old embedded material is already carried by the graph edges.
            external[position] = (
                _WASH_REMAINING[_PA_FIBRE_GRADES[material, constituent]] * mass[node]
            )
        elif stage == "discard":
            weights = cohort_weights[region, material, good]
            entries = history.get((region, material, constituent, good), {})
            external[position] = mass[node] * fsum(
                weight * entries[int(cohort)]
                for cohort, weight in zip(weights.prior_years, weights.prior_weights, strict=True)
            )
    edges = [edge for edge in graph.edges if edge.source in positions]
    coefficients = []
    for edge in edges:
        fraction = edge.numerator / mass[edge.source]
        if edge.source[0] == "manufacturing_input":
            deposited = deposit_fractions[edge.source[2], edge.source[3]]
            if edge.target[0] == "processing":
                fraction = 1 - deposited
            elif edge.target[0] == "spinning_equipment_deposits":
                fraction = deposited
            else:
                # The other PA manufacturing exits are selective caprolactam.
                fraction = 0.0
        coefficients.append(fraction)
    transfer = cast(
        Any,
        coo_matrix(
            (
                coefficients,
                ([positions[e.target] for e in edges], [positions[e.source] for e in edges]),
            ),
            shape=(len(nodes), len(nodes)),
        ),
    ).tocsc()
    # Only the two PA systems are solved. Actual recovery and cohort fractions
    # determine same-year feedback.
    system = (eye(len(nodes), format="csc") - transfer).tocsc()
    amounts = splu(system).solve(external)
    reported: PAFinishAmounts = {}
    for edge, fraction in zip(edges, coefficients, strict=True):
        meta = edge.metadata
        if meta is not None and meta.flow in _REPORTED_FLOWS:
            reported[year, meta.region, meta.material, meta.constituent, meta.good, meta.flow] = (
                float(amounts[positions[edge.source]]) * fraction
            )
    return {node: float(amounts[i]) for node, i in positions.items()}, reported


def prepare_carbon_cohort_weights(
    cohort_state: pd.DataFrame,
    cohort_material_identity: pd.DataFrame,
) -> CohortWeightsByYear:
    """Weight each year's material discards by their contributing cohort masses.

    Multiply cohort outflows by the material shares at entry, then normalize
    within event year, region, material and use. Preserve cohort order and
    separate prior entries from the current year's entry for carbon tracing.
    """

    material_identity = cohort_material_identity[
        ["cohort_year", "region", "material", "good", "material_share"]
    ].set_index(["cohort_year", "region", "good"])
    outflows = cohort_state.loc[
        (cohort_state["stock_quantity"] == "outflow")
        & (cohort_state["amount"] > 0)
        & (cohort_state["event_year"] <= _REPORTING_END_YEAR),
        ["cohort_year", "event_year", "region", "good", "amount"],
    ]
    annual = outflows.merge(
        material_identity,
        left_on=["cohort_year", "region", "good"],
        right_index=True,
        how="inner",
        sort=False,
    ).reset_index(drop=True)
    annual["amount_mt"] = annual["amount"] * annual["material_share"]
    annual = annual.loc[annual["amount_mt"] > 0]
    totals = annual.groupby(
        ["event_year", "region", "material", "good"],
        sort=False,
        observed=True,
    )["amount_mt"].transform("sum")
    annual["weight"] = annual["amount_mt"] / totals

    cohort_years = annual["cohort_year"].to_numpy(dtype=int, copy=False)
    weight_values = annual["weight"].to_numpy(dtype=float, copy=False)
    groups = annual.groupby(
        ["event_year", "region", "material", "good"],
        sort=False,
        observed=True,
    ).indices
    weights: CohortWeightsByYear = {}
    for coordinates, positions in groups.items():
        event_year, region, material, good = cast(tuple[int, str, str, str], coordinates)
        years = cohort_years[positions]
        values = weight_values[positions]
        prior = years < int(event_year)
        current = years == int(event_year)
        current_weight = 0.0 if not np.any(current) else float(values[current][0])
        weights.setdefault(int(event_year), {})[(region, material, good)] = _CohortDiscardWeights(
            prior_years=years[prior],
            prior_weights=values[prior],
            current_weight=current_weight,
        )
    return weights


def virgin_route_shares(nominal: pd.DataFrame) -> dict[tuple[int, str, str], np.ndarray]:
    """Read common virgin-route ratios; Mt polymer-equivalent magnitudes are unused.

    A zero virgin-material requirement gives zero virgin carbon and LCI demand.
    """
    table = nominal.pivot(
        index=["year", "region", "material"], columns="feedstock", values="amount"
    )
    values = table[["fossil", "biogenic", "daccu"]].to_numpy()
    totals = np.asarray([fsum(row) for row in values])
    shares = np.divide(
        values, totals[:, None], out=np.zeros_like(values), where=totals[:, None] > 0
    )
    return dict(zip(table.index, shares, strict=True))


def _history_entries(
    history: EntryHistory,
    tps_constituents: frozenset[tuple[str, str]],
    pa_history: PAFinishEntryHistory,
) -> CarbonCustodyHistory:
    """Capture carbon and TPS route densities in stable cohort/constituent/use order."""
    rows = tuple(
        sorted(
            (year, *coordinate, float(value[0]), float(value[1]), float(value[2]))
            for coordinate, years in history.items()
            for year, value in years.items()
        )
    )
    return CarbonCustodyHistory(
        _REPORTING_START_YEAR,
        tuple(row for row in rows if (row[2], row[3]) not in tps_constituents),
        tuple(row for row in rows if (row[2], row[3]) in tps_constituents),
        tuple(
            sorted(
                (year, *coordinate, density)
                for coordinate, years in pa_history.items()
                for year, density in years.items()
            )
        ),
    )


def _entry_history(history: CarbonCustodyHistory | None) -> EntryHistory:
    """Restore the immutable entry-density rows used by prospective candidates."""
    entries: EntryHistory = {}
    if history is not None:
        for year, region, material, constituent, good, first, second, third in (
            *history.entries,
            *history.tps_route_entries,
        ):
            entries.setdefault((region, material, constituent, good), {})[year] = np.array(
                (first, second, third)
            )
    return entries


def _discard_origins(
    amount: float,
    weights: _CohortDiscardWeights,
    history: dict[int, np.ndarray],
) -> tuple[float, np.ndarray, float]:
    """Multiply prior discarded constituent masses by their actual entry densities."""
    prior_mass = amount * fsum(weights.prior_weights)
    parts = [
        amount * weight * history[int(year)]
        for year, weight in zip(weights.prior_years, weights.prior_weights, strict=True)
    ]
    prior_origins = np.array([fsum(part[origin] for part in parts) for origin in range(3)])
    return prior_mass, prior_origins, amount * weights.current_weight


def _material_amounts(table: pd.DataFrame) -> dict[tuple[str, str, str, str, str], float]:
    """Index one admitted year's constituent quantities with their source/destination owner."""
    material = table["destination_material"].fillna(table["source_material"])
    constituent = table["destination_constituent"].fillna(table["source_constituent"])
    return dict(
        zip(
            zip(
                table["flow_name"],
                table["region"],
                material,
                constituent,
                table["product"],
                strict=True,
            ),
            table["amount"],
            strict=True,
        )
    )


def _chemical_carbon(
    composition: tuple[MaterialConstituent, ...],
    material_carbon: MaterialCarbon,
    recipe: EcoinventMaterialRecipe,
) -> _ChemicalCarbon:
    """Derive backbone carbon and monomer retention from the prepared authorities.

    For each eligible constituent, multiply its ethylene and propylene needs
    (kg monomer/kg polymer) by their carbon contents (kg C/kg monomer). Divide
    prepared backbone carbon (kg C/kg polymer) by that sum to obtain the
    fraction of supplied monomer carbon retained in the polymer.
    These recipe and carbon coefficients are fixed across scenarios.
    """
    monomers = {
        p.product: p.carbon_kg_per_kg for p in recipe.cracker_products if p.product in _MONOMERS
    }
    requirements = {
        p.polymer: (p.ethylene_kg_per_kg, p.propylene_kg_per_kg) for p in recipe.polymers
    }
    backbone = np.zeros(len(composition))
    retention = np.zeros(len(composition))
    for index, item in enumerate(composition):
        if item.chemical_polymer is None:
            continue
        density = fsum(material_carbon.fractions[index][0])
        gross = fsum(
            monomers[p] * amount
            for p, amount in zip(_MONOMERS, requirements[item.chemical_polymer], strict=True)
        )
        backbone[index], retention[index] = density, density / gross
    return _ChemicalCarbon(monomers, backbone, retention)


def _manufacturing_carbon(
    composition: tuple[MaterialConstituent, ...],
    carbon: np.ndarray,
    recipe: EcoinventMaterialRecipe,
) -> tuple[_ManufacturingCarbon, ...]:
    """Partition incoming principal carbon using the admitted manufacturing exits.

    Source exits are kg per finished kg. Styrene, butadiene and MMA withdraw
    their molecular carbon before the specified subsequent homogeneous loss.
    UPR and SBR trim after release; road marking leaves container residue first.
    These three selected constituents have one common origin for their principal
    components and no recovered-feed entry. Their imported origin mixtures can
    therefore use the same scalar coefficient.
    Ordinary losses retain the incoming carbon composition, including epoxy
    overspray; compound auxiliaries remain outside this principal account.

    An exit's incoming-principal-equivalent mass is its mass times
    its carbon fraction divided by the incoming principal carbon density.
    Multiplying this equivalent mass by that density later recovers the actual
    carbon loss. Exits and their source equations are owned by
    ``ecoinvent_materials._upr_conversion`` and ``_reconstructed_conversions``.
    """
    requirements = {row.operation: row for row in recipe.conversion}
    result = []
    for index, item in enumerate(composition):
        density = fsum(carbon[index, 0])
        finished_parts = []
        physical_parts: dict[str, list[float]] = defaultdict(list)
        carbon_parts: dict[str, list[float]] = defaultdict(list)
        for component in item.conversion.components:
            source = requirements[component.operation]
            gross = source.principal_feed_kg_per_finished_kg
            physical = {exit.name: exit.kg_per_finished_kg for exit in source.principal_exits}
            equivalents = physical.copy()
            if component.operation == "others_upr_hand_lay_up_conversion":
                styrene = physical["styrene_air"]
                # Nominal C=12, H=1: styrene C8H8 contains 96/104 = 12/13 kg C/kg.
                equivalents["styrene_air"] = styrene * (12 / 13) / density
                # Trim follows styrene release. Divide remaining equivalent by
                # remaining principal to express trim on the incoming basis.
                equivalents["post_release_solid"] = (
                    physical["post_release_solid"]
                    * (gross - equivalents["styrene_air"])
                    / (gross - styrene)
                )
                retained = gross - fsum(equivalents.values())
            elif component.operation == "others_esbr_compounding_conversion":
                styrene, butadiene = physical["styrene_air"], physical["butadiene_air"]
                # The same nominal masses give C8H8: 96/104 = 12/13 and
                # butadiene C4H6: 48/54 = 8/9 kg C per kg of released monomer.
                equivalents["styrene_air"] = styrene * (12 / 13) / density
                equivalents["butadiene_air"] = butadiene * (8 / 9) / density
                dust = physical["extrusion_dust"]
                # Homogeneous charge dust has equal actual mass and source-equivalent mass.
                # Subtract it and both monomers before allocating the later trim.
                equivalents["post_release_solid"] = (
                    physical["post_release_solid"]
                    * (gross - equivalents["styrene_air"] - equivalents["butadiene_air"] - dust)
                    / (gross - styrene - butadiene - dust)
                )
                retained = gross - fsum(equivalents.values())
            elif component.operation == "others_road_marking_conversion":
                # With O=16, methyl methacrylate C5H8O2 contains 60/100 = 3/5 kg C/kg.
                # Homogeneous container residue has already left before this release.
                equivalents["methyl_methacrylate_air"] = (
                    physical["methyl_methacrylate_air"] * (3 / 5) / density
                )
                retained = gross - fsum(equivalents.values())
            else:
                # One finished kg keeps the incoming density for homogeneous losses;
                # this is a carbon-equivalent coefficient.
                retained = 1.0
            weight = float(component.share)
            finished_parts.append(weight * retained)
            for name, amount in physical.items():
                physical_parts[name].append(weight * amount)
                carbon_parts[name].append(weight * equivalents[name])
        result.append(
            _ManufacturingCarbon(
                fsum(finished_parts),
                tuple(
                    _PrincipalCarbonExit(name, fsum(parts), fsum(carbon_parts[name]))
                    for name, parts in physical_parts.items()
                ),
            )
        )
    return tuple(result)


def _build_year_graph(
    year: int,
    flows: pd.Series,
    material_table: pd.DataFrame,
    composition: tuple[MaterialConstituent, ...],
    injection_basis: np.ndarray,
    chemistry: _ChemicalCarbon,
    manufacturing: tuple[_ManufacturingCarbon, ...],
    shares: dict[tuple[int, str, str], np.ndarray],
    cohort_weights: dict[tuple[str, str, str], _CohortDiscardWeights],
    history: EntryHistory,
) -> tuple[_YearGraph, dict[Node, tuple[str, str, str, str]]]:
    """Connect markets, constituent treatments and regional monomer pools.

    ``injection_basis`` maps new route quantities to ultimate carbon for non-TPS
    constituents and is the identity matrix for TPS principal mass. All later
    transfers use the constituent's same basis, including historical discards.
    """
    graph = _YearGraph()
    entries = {}
    amounts = flows.to_dict()
    materials = _material_amounts(material_table)
    regions = tuple(flows.index.get_level_values("region").unique())
    goods = tuple(g for g in flows.index.get_level_values("good").unique() if g != "all")
    monomer_carbon = chemistry.monomers

    for region in regions:
        for product in _MONOMERS:
            pool = ("monomer", region, product, "all", "all")
            supply = fsum(
                materials[("cracker_product", region, c.material, c.constituent, product)]
                for c in composition
            )
            if supply > 0:
                graph.mass_inputs[pool].append(supply)
        for index, item in enumerate(composition):
            material, constituent = item.material, item.constituent

            def node(stage: str, *, good: str = "all", global_market: bool = False) -> Node:
                """Keep constituent/use coordinates on local and global process nodes."""
                return stage, "all" if global_market else region, material, constituent, good

            def family(flow: str, good: str = "all") -> float:
                """Project an admitted family polymer flow to its fixed constituent mass."""
                return float(item.share) * amounts[(region, material, good, flow)]

            def quantity(flow: str, product: str = "polymer") -> float:
                """Read a selective treatment or new-polymer quantity."""
                return materials[(flow, region, material, constituent, product)]

            def physical(flow: str, amount: float, *, good: str = "all") -> None:
                """Connect an ordinary polymer flow and retain meaningful report coordinates."""
                source, target = flow.split(" => ")
                graph.transfer(
                    node(
                        source,
                        good=good if source in {"use", "good_market"} else "all",
                        global_market=source.endswith("_market"),
                    ),
                    node(
                        "manufacturing_input" if target == "processing" else target,
                        good=good if target in {"use", "good_market"} else "all",
                        global_market=target.endswith("_market"),
                    ),
                    amount,
                    _FlowMetadata(region, material, constituent, good, flow),
                )

            production = quantity("new_polymer_domestic") + quantity("new_polymer_export")
            route_shares = shares[(year, region, material)]
            if item.chemical_polymer is None:
                new_carbon = production * (route_shares @ injection_basis[index])
            else:
                retained_carbon = chemistry.backbone[index]
                retention = chemistry.retention[index]
                virgin_carbon = retention * fsum(
                    monomer_carbon[p] * quantity("virgin_monomer_consumption", p) for p in _MONOMERS
                )
                new_carbon = virgin_carbon * route_shares
                for product in _MONOMERS:
                    pool = ("monomer", region, product, "all", "all")
                    recovered = quantity("recovered_monomer_consumption", product)
                    graph.transfer(
                        pool,
                        node("new_polymer"),
                        retention * recovered,
                        _FlowMetadata(
                            region,
                            material,
                            constituent,
                            "all",
                            "recovered_monomer_consumption",
                            product,
                        ),
                        physical_amount=0.0,
                    )
                    produced = quantity("cracker_product", product)
                    graph.transfer(
                        node("chemical_feedstock"),
                        pool,
                        monomer_carbon[product] * produced / retained_carbon,
                        physical_amount=0.0,
                    )
            graph.inject(node("new_polymer"), production, new_carbon)

            for flow, field in (
                ("new_polymer => processing", "new_polymer_domestic"),
                ("new_polymer => primary_market", "new_polymer_export"),
                ("collected => reclmech", "mechanical_treatment"),
                ("collected => reclchem", "chemical_treatment"),
                ("collected => incineration", "incineration_treatment"),
                ("collected => landfill", "landfill_treatment"),
                ("reclmech => processing", "mechanical_resin"),
                ("reclmech => incineration", "mechanical_incinerated_reject"),
                ("reclmech => landfill", "mechanical_landfilled_reject"),
                ("reclmech => waterborne_loss", "mechanical_waterborne_loss"),
                ("reclmech => airborne_loss", "mechanical_airborne_loss"),
                ("reclchem => chemical_feedstock", "chemical_processed_feed"),
                ("reclchem => chemical_preparation_reject", "chemical_preparation_reject"),
            ):
                physical(flow, quantity(field))
            for flow in (
                "primary_market => processing",
                "processing => fabrication",
                "processing => intermediate_market",
                "intermediate_market => fabrication",
                "eol => collected",
                "eol => mismanaged",
                "collected => waste_market",
                "waste_market => collected",
            ):
                physical(flow, family(flow))
            finished = quantity("manufacturing_finished_output")
            retention = manufacturing[index]
            graph.transfer(
                node("manufacturing_input"),
                node("processing"),
                finished * retention.finished_equivalent,
                _FlowMetadata(
                    region, material, constituent, "all", "processing => finished_principal"
                ),
                physical_amount=finished,
            )
            for exit in retention.exits:
                graph.transfer(
                    node("manufacturing_input"),
                    node(exit.name),
                    finished * exit.carbon_equivalent,
                    _FlowMetadata(
                        region, material, constituent, "all", f"processing => {exit.name}"
                    ),
                    physical_amount=finished * exit.physical,
                )
            for good in goods:
                for flow in (
                    "fabrication => use",
                    "fabrication => good_market",
                    "good_market => use",
                ):
                    physical(flow, family(flow, good), good=good)
                use = node("use", good=good)
                entries[use] = (region, material, constituent, good)
                discard = node("discard", good=good)
                amount = family("use => eol", good)
                if amount > 0:
                    weights = cohort_weights[(region, material, good)]
                    prior, origin, current = _discard_origins(
                        amount, weights, history.get((region, material, constituent, good), {})
                    )
                    graph.inject(discard, prior, origin)
                    graph.transfer(use, discard, current)
                    graph.transfer(discard, node("eol"), amount)
    return graph, entries


def _custody_rows(
    scenario: str,
    year: int,
    graph: _YearGraph,
    amounts: np.ndarray,
    tps_carbon: dict[tuple[str, str], np.ndarray],
) -> tuple[list[dict[str, object]], TPSRouteAmounts]:
    """Report Mt C by ultimate origin and retain TPS route Mt for inventory construction.

    TPS source-route masses become carbon only through the prepared forward
    matrix. Its synthetic routes retain their biogenic primer carbon separately
    from the native-starch formulation.
    """
    rows: list[dict[str, object]] = []
    tps_routes: TPSRouteAmounts = {}
    for edge, carbon in zip(graph.edges, amounts, strict=True):
        meta = edge.metadata
        if meta is None or meta.flow not in _REPORTED_FLOWS:
            continue
        coordinate = (meta.material, meta.constituent)
        if coordinate in tps_carbon:
            tps_routes[(year, meta.region, *coordinate, meta.good, meta.flow)] = (
                float(carbon[0]),
                float(carbon[1]),
                float(carbon[2]),
            )
            carbon = carbon @ tps_carbon[coordinate]
        total = fsum(carbon)
        for origin, amount in zip(ULTIMATE_CARBON_ORIGINS, carbon, strict=True):
            if amount > 0:
                rows.append(
                    dict(
                        foreground_scenario=scenario,
                        year=year,
                        region=meta.region,
                        material=meta.material,
                        constituent=meta.constituent,
                        good=meta.good,
                        source_flow_name=meta.flow,
                        product=meta.product,
                        custody_status=_REPORTED_FLOWS[meta.flow],
                        ultimate_carbon_origin=origin,
                        carbon_amount_mt=float(amount),
                        source_carbon_amount_mt=total,
                        origin_fraction=float(amount / total),
                        unit="Mt C",
                    )
                )
    return rows, tps_routes


def trace_carbon_custody(
    mfa: MFAEvaluation | MFACandidateEvaluation,
    material_carbon: MaterialCarbon,
    recipe: EcoinventMaterialRecipe,
    *,
    history: CarbonCustodyHistory | None = None,
    cohort_weights_by_year: CohortWeightsByYear | None = None,
) -> CarbonCustodyTrace:
    """Trace admitted flows and retain end-2019 constituent entry densities.

    Args:
        mfa: Admitted tables; the two nominal elements are summed to polymer mass.
        material_carbon: Prepared kg C/kg constituent by virgin route and ultimate origin.
        recipe: The same identity-admitted monomer recipe used by the MFA owner.
        history: Optional baseline cohort densities; candidates calculate only later years.
        cohort_weights_by_year: Optional demand/lifetime discard weights, reused
            by candidates.

    Returns:
        Constituent-carbon amounts for 2019--2100 and the historical entry densities.
        Native inventory carbon and MFA nominal atmospheric bookkeeping are separate.
    """
    start = 1950 if history is None else history.through_year + 1
    selected = mfa.model_flows.loc[mfa.model_flows["year"].between(start, _REPORTING_END_YEAR)]
    scenario = str(mfa.model_flows["foreground_scenario"].iloc[0])
    flows = selected.groupby(
        ["year", "region", "material", "good", "source_flow_name"], observed=True, sort=True
    )["amount"].sum()
    material_groups = mfa.material_realization.groupby("year", sort=False).indices
    composition = load_material_composition().constituents
    carbon = np.asarray(material_carbon.fractions)
    chemistry = _chemical_carbon(composition, material_carbon, recipe)
    manufacturing = _manufacturing_carbon(composition, carbon, recipe)
    tps_carbon = {
        (item.material, item.constituent): carbon[index]
        for index, item in enumerate(composition)
        if item.mechanical_recovery == "tps_compound"
    }
    tps_constituents = frozenset(tps_carbon)
    injection_basis = carbon.copy()
    for index, item in enumerate(composition):
        if (item.material, item.constituent) in tps_constituents:
            injection_basis[index] = np.eye(3)
    shares = virgin_route_shares(mfa.nominal_feedstock_equivalents)
    if cohort_weights_by_year is None:
        cohort_weights_by_year = prepare_carbon_cohort_weights(
            mfa.cohort_state, mfa.cohort_material_identity
        )
    entries = _entry_history(history)
    pa_entries: PAFinishEntryHistory = {}
    if history is not None:
        for year, region, material, constituent, good, density in history.pa_finish_entries:
            pa_entries.setdefault((region, material, constituent, good), {})[year] = density
    # Principal recipe admission owns the deposit quantity. Derive its input
    # fraction here rather than repeat the EIA coefficient or a rounded yield.
    requirements = {item.operation: item for item in recipe.conversion}
    deposit_fractions = {
        coordinate: next(
            exit.kg_per_finished_kg
            for exit in requirements[f"fibres_{grade}_conversion"].principal_exits
            if exit.name == "spinning_equipment_deposits"
        )
        / requirements[f"fibres_{grade}_conversion"].principal_feed_kg_per_finished_kg
        for coordinate, grade in _PA_FIBRE_GRADES.items()
    }
    captured = (
        CarbonCustodyHistory(_REPORTING_START_YEAR, (), (), ()) if history is None else history
    )
    rows = []
    tps_route_amounts: TPSRouteAmounts = {}
    pa_finish_amounts: PAFinishAmounts = {}
    for year, annual in flows.groupby(level="year", sort=True):
        year_int = int(cast(int, year))
        graph, entry_nodes = _build_year_graph(
            year_int,
            annual.droplevel("year"),
            mfa.material_realization.iloc[material_groups[year_int]],
            composition,
            injection_basis,
            chemistry,
            manufacturing,
            shares,
            cohort_weights_by_year.get(year_int, {}),
            entries,
        )
        node_amounts, edge_amounts, mass = _solve_origin_mixing_network(
            graph, year=year_int, tps_constituents=tps_constituents
        )
        pa_node_amounts, pa_amounts = _solve_pa_finish_custody(
            graph,
            mass,
            year_int,
            cohort_weights_by_year.get(year_int, {}),
            pa_entries,
            deposit_fractions,
        )
        if year_int >= _REPORTING_START_YEAR:
            pa_finish_amounts.update(pa_amounts)
            annual_rows, tps_routes = _custody_rows(
                scenario, year_int, graph, edge_amounts, tps_carbon
            )
            rows.extend(annual_rows)
            tps_route_amounts.update(tps_routes)
        for node, coordinate in entry_nodes.items():
            if node in mass:
                entries.setdefault(coordinate, {})[year_int] = node_amounts[node] / mass[node]
                if node in pa_node_amounts:
                    pa_entries.setdefault(coordinate, {})[year_int] = (
                        pa_node_amounts[node] / mass[node]
                    )
        if history is None and year_int == _REPORTING_START_YEAR:
            captured = _history_entries(entries, tps_constituents, pa_entries)
    table = pd.DataFrame(rows, columns=_CUSTODY_COLUMNS).sort_values(
        [
            "year",
            "region",
            "material",
            "constituent",
            "good",
            "source_flow_name",
            "product",
            "ultimate_carbon_origin",
        ],
        kind="stable",
        ignore_index=True,
    )
    return CarbonCustodyTrace(
        table,
        captured,
        chemistry,
        max(start, _REPORTING_START_YEAR),
        tps_route_amounts,
        pa_finish_amounts,
    )
