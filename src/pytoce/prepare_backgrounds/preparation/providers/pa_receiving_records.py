"""Native PA receiving records and their serialization.

These records preserve native units, signed source production and
normalized exchange metadata across thermal, gas and water admission. Completed package records
supply the admitted data for subsequent calculations.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isnan
from typing import cast

from .recipe_context import _scaled_native_exchange

_ROW_METADATA = (
    "uncertainty type",
    "loc",
    "scale",
    "shape",
    "minimum",
    "maximum",
    "negative",
    "allocation",
    "formula",
    "activity",
    "flow",
)


@dataclass(frozen=True, slots=True)
class PaReceivingActivity:
    """One native or prepared provider product."""

    key: tuple[str, str]
    name: str
    reference_product: str
    unit: str
    location: str


@dataclass(frozen=True, slots=True)
class PaReceivingFlow:
    """An existing elementary flow with its exact unit and compartment."""

    key: tuple[str, str]
    name: str
    unit: str
    categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PaReceivingRow:
    """One native row per positive source-product unit, in its endpoint's unit.

    Metadata retains normalized uncertainty and native allocation/flow provenance.
    NaN markers for absent scale, shape or bounds are omitted: the native formatter restores their
    missing defaults and preserves the distribution.
    Rows with a common product retain their distinct supplier identities. Their
    coefficients may be aggregated for chemistry, but their supplies must still
    be allocated to the actual individual endpoints by the receiving owner.
    """

    endpoint: PaReceivingActivity | PaReceivingFlow
    amount: float
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class PaReceivingOperation:
    """A source identity, signed native production and already normalized rows.

    ``production_amount`` preserves the native sign and source basis.
    The rows are normalized once by the absolute reference-production magnitude.
    """

    activity: PaReceivingActivity
    production_amount: float
    rows: tuple[PaReceivingRow, ...]

    def supplies(self) -> dict[str, float]:
        """Sum normalized product coefficients grouped by native name."""
        amounts: dict[str, float] = {}
        for row in self.rows:
            if isinstance(row.endpoint, PaReceivingActivity):
                product = row.endpoint.reference_product
                amounts[product] = amounts.get(product, 0.0) + row.amount
        return amounts

    def releases(self) -> dict[tuple[str, tuple[str, ...]], float]:
        """Sum native elementary coefficients by name and actual compartment."""
        amounts: dict[tuple[str, tuple[str, ...]], float] = {}
        for row in self.rows:
            if isinstance(row.endpoint, PaReceivingFlow):
                key = row.endpoint.name, row.endpoint.categories
                amounts[key] = amounts.get(key, 0.0) + row.amount
        return amounts


@dataclass(frozen=True, slots=True)
class PaReceivingLandfarmingReference:
    """Original soil/water/air coefficients and intensive sludge denominators.

    Dry and water fractions are kg/kg wet sludge; contents are kg element/kg
    dry sludge. Each receiving owner selects the required source coefficients
    and its phosphorus calculation.
    """

    operation: PaReceivingOperation
    dry_fraction: float
    water_fraction: float
    element_contents: dict[str, float]


@dataclass(frozen=True, slots=True)
class PaReceivingAshReference:
    """Original ash release coefficients and kg-element/kg-dry denominators.

    Dry fraction is kg dry/kg received treatment reference. The operation keeps
    selected native short/long release factors and uncertainties. Dated cement,
    hydration, capital and haulage belong to the receiving source operations.
    """

    operation: PaReceivingOperation
    dry_fraction: float
    element_contents: dict[str, float]


def _activity(node: Mapping[str, object]) -> PaReceivingActivity:
    """Detach the native database/product identity from a selected source node."""
    return PaReceivingActivity(
        key=(str(node["database"]), str(node["code"])),
        name=str(node["name"]),
        reference_product=str(node["reference product"]),
        unit=str(node["unit"]),
        location=str(node["location"]),
    )


def _flow(node: Mapping[str, object]) -> PaReceivingFlow:
    """Detach the selected biosphere endpoint with its original attributes."""
    return PaReceivingFlow(
        key=(str(node["database"]), str(node["code"])),
        name=str(node["name"]),
        unit=str(node["unit"]),
        categories=tuple(cast(tuple[str, ...], node["categories"])),
    )


def _activity_from_payload(payload: Mapping[str, object]) -> PaReceivingActivity:
    """Restore the admitted product identity, including its database/code tuple."""
    return PaReceivingActivity(
        key=cast(tuple[str, str], tuple(cast(list[str], payload["key"]))),
        name=str(payload["name"]),
        reference_product=str(payload["reference_product"]),
        unit=str(payload["unit"]),
        location=str(payload["location"]),
    )


def _flow_from_payload(payload: Mapping[str, object]) -> PaReceivingFlow:
    """Restore the admitted elementary identity and compartment tuple."""
    return PaReceivingFlow(
        key=cast(tuple[str, str], tuple(cast(list[str], payload["key"]))),
        name=str(payload["name"]),
        unit=str(payload["unit"]),
        categories=tuple(cast(list[str], payload["categories"])),
    )


def _operation_from_payload(payload: Mapping[str, object]) -> PaReceivingOperation:
    """Restore native row order, typed endpoints and already normalized amounts."""
    rows: list[PaReceivingRow] = []
    for row in cast(list[dict[str, object]], payload["rows"]):
        endpoint = cast(dict[str, object], row["endpoint"])
        rows.append(
            PaReceivingRow(
                endpoint=(
                    _activity_from_payload(endpoint)
                    if "reference_product" in endpoint
                    else _flow_from_payload(endpoint)
                ),
                amount=float(cast(float, row["amount"])),
                metadata=cast(dict[str, object], row["metadata"]),
            )
        )
    return PaReceivingOperation(
        activity=_activity_from_payload(cast(dict[str, object], payload["activity"])),
        production_amount=float(cast(float, payload["production_amount"])),
        rows=tuple(rows),
    )


def _ash_reference_from_payload(payload: Mapping[str, object]) -> PaReceivingAshReference:
    """Restore trusted original ash calibration and its intensive units."""
    return PaReceivingAshReference(
        _operation_from_payload(cast(Mapping[str, object], payload["operation"])),
        float(cast(float, payload["dry_fraction"])),
        dict(cast(Mapping[str, float], payload["element_contents"])),
    )


def _landfarming_reference_from_payload(
    payload: Mapping[str, object],
) -> PaReceivingLandfarmingReference:
    """Restore trusted original sludge-to-soil calibration and its intensive units."""
    return PaReceivingLandfarmingReference(
        _operation_from_payload(cast(Mapping[str, object], payload["operation"])),
        float(cast(float, payload["dry_fraction"])),
        float(cast(float, payload["water_fraction"])),
        dict(cast(Mapping[str, float], payload["element_contents"])),
    )


def _receiving_row(
    endpoint: PaReceivingActivity | PaReceivingFlow,
    native: Mapping[str, object],
    multiplier: float,
) -> PaReceivingRow:
    """Normalize one admitted row while preserving its portable native metadata.

    The native operation owner supplies the row selection and scale.
    bw_processing 1.6 defaults absent scale, shape and bounds to NaN; omitting
    their missing markers gives portable JSON that preserves the native distribution. Keep the
    normalized loc, other defined values and false flags.
    """
    normalized = _scaled_native_exchange(native, multiplier)
    return PaReceivingRow(
        endpoint=endpoint,
        amount=float(cast(float, normalized["amount"])),
        metadata={
            field: normalized[field]
            for field in _ROW_METADATA
            if field in normalized
            and not (
                field in {"scale", "shape", "minimum", "maximum"}
                and isnan(cast(float, normalized[field]))
            )
        },
    )
