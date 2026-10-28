"""Provider-route sources and the licensed open-dump portfolio calculation."""

import csv
import json
from importlib.resources import files
from pathlib import Path
from typing import cast

import pandas as pd

OPEN_DUMP_PORTFOLIO_SOURCE_ID = "ecoinvent_3_12_row_open_dump_portfolio"

_DIRECT_TREATMENT_CARBON_SPECIES = (
    ("Carbon dioxide, fossil", "Carbon dioxide, non-fossil"),
    ("Methane, fossil", "Methane, non-fossil"),
    ("Carbon monoxide, fossil", "Carbon monoxide, non-fossil"),
)


def _direct_carbon_mappings(
    categories: tuple[str, ...],
    *,
    species: tuple[tuple[str, str], ...] = _DIRECT_TREATMENT_CARBON_SPECIES,
) -> list[dict[str, object]]:
    """Return the exact eligible direct waste-carbon identity mappings."""

    return [
        {
            "selector": {
                "name": fossil,
                "unit": "kilogram",
                "categories": list(categories),
            },
            "replacement_selector": {
                "name": non_fossil,
                "unit": "kilogram",
                "categories": list(categories),
            },
            "attribution_basis": (
                "The exact treatment activity attributes this direct carbon flow "
                "to the treated plastic input."
            ),
            "carbon_attribution_status": "separable_treated_plastic_carbon",
            "mapping_status": "exact_non_fossil_counterpart",
            "reason": (
                "Preserve the native amount and compartment while representing "
                "non-fossil plastic carbon."
            ),
            "limitation": (
                "Only the direct treated-plastic carbon-flow identity changes; "
                "all other native exchanges and metadata remain unchanged."
            ),
        }
        for fossil, non_fossil in species
    ]


def _provider_resource_path(name: str) -> Path:
    """Locate one packaged provider recipe resource."""
    return Path(str(files("pytoce.prepare_backgrounds.data.provider_recipes").joinpath(name)))


def _chemical_recycling_authority() -> dict[str, object]:
    """Read the chemical recycling source and method record."""
    authority = cast(
        dict[str, object],
        json.loads(
            _provider_resource_path("chemical_recycling_authority.json").read_text(encoding="utf-8")
        ),
    )
    return authority


def _sbr_treatment_authority() -> dict[str, object]:
    """Read the SBR treatment source and method record."""
    return cast(
        dict[str, object],
        json.loads(
            _provider_resource_path("sbr_treatment_authority.json").read_text(encoding="utf-8")
        ),
    )


def _hybrid_provider_temporal_authority() -> dict[str, object]:
    """Read the early-year provider and background time mapping."""
    return cast(
        dict[str, object],
        json.loads(
            _provider_resource_path("hybrid_provider_temporal_authority.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def _csv_rows(name: str) -> tuple[dict[str, str], ...]:
    """Read one provider resource CSV as string-valued records."""
    with _provider_resource_path(name).open(encoding="utf-8", newline="") as source:
        return tuple(dict(row) for row in csv.DictReader(source))


def mismanaged_fate_authorities() -> pd.DataFrame:
    """Return the common open-burning and open-dumping split for all 12 world regions."""

    fate = pd.DataFrame(_csv_rows("h12_mismanaged_eol_fate_shares.csv"))
    for column in ("open_burning_share", "open_dump_share", "share_sum"):
        fate[column] = fate[column].astype(float)
    return fate


def resolve_open_dump_routes(
    source_database: str,
    routes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Derive the common open-dump class shares once from the licensed PE market.

    The conditional class weight is ``abs(a_i) / sum(abs(a_j))`` for the five
    declared open-dump inputs, rounded to fifteen significant digits. Other
    treatments and transport are outside this denominator. The market's
    production divisor cancels in this ratio. Multiply each material
    component share by its class weight. Native exchanges retain their original quantities.

    The fixed RoW portfolio applies uniformly to every region and year. Call after fresh ecoinvent
    import,
    and save the returned routes and source details in the shared product.
    """
    import bw2data as bd

    classes = {
        row["activity_name"]: row["infiltration_class"]
        for row in _csv_rows("provider_selector_catalogue.csv")
        if row["fate"] == "open_dump" and row["provider_service_family"] == "PE"
    }
    market = bd.get_activity(
        database=source_database,
        name="market for waste polyethylene",
        product="waste polyethylene",
        location="RoW",
        unit="kilogram",
    )
    amounts: dict[str, float] = {}
    for exchange in market.technosphere():
        name = str(exchange.input["name"])
        if name in classes:
            amounts[classes[name]] = float(exchange["amount"])
    denominator = sum(abs(amounts[name]) for name in classes.values())
    weights = {
        name: float(format(abs(amounts[name]) / denominator, ".15g")) for name in classes.values()
    }
    resolved = routes.copy()
    dumping = resolved["provider_demand_type"] == "mismanaged_open_dump_treatment"
    resolved.loc[dumping, "infiltration_share"] = resolved.loc[dumping, "infiltration_class"].map(
        weights
    )
    resolved.loc[dumping, "share"] = resolved.loc[dumping, "material_component_share"].astype(
        float
    ) * resolved.loc[dumping, "infiltration_share"].astype(float)
    source: dict[str, object] = {
        "source_id": OPEN_DUMP_PORTFOLIO_SOURCE_ID,
        "native_source_key_json": json.dumps(market.key, separators=(",", ":")),
        "derivation_json": json.dumps(
            {
                "native_exchange_amounts": amounts,
                "conditional_denominator": denominator,
                "class_shares": weights,
                "equation": "abs(a_i)/sum(abs(a_j)), fifteen significant digits",
                "applicability": "Common RoW class portfolio for every world region and year",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    return resolved, source
