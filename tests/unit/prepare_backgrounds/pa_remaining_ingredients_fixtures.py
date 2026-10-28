"""Synthetic fatty alcohol sulfate delivery and polyamide phosphate inventories."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from material_recipe_fixtures import physical_material_recipe
from native_chemical_fixtures import native_chemical_biosphere, native_chemical_inventory

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)

# The selected market names eight fatty-alcohol-sulfate suppliers. Quantities are synthetic test
# inputs.
FAS_SUPPLIERS = tuple(
    (
        f"fas_{origin}_{location.lower()}",
        name,
        location,
    )
    for origin, name in (
        ("coconut", "fatty alcohol sulfate production, coconut oil"),
        ("palm", "fatty alcohol sulfate production, palm oil"),
        ("palm_kernel", "fatty alcohol sulfate production, from palm kernel oil"),
        ("petrochemical", "fatty alcohol sulfate production, petrochemical"),
    )
    for location in ("RER", "RoW")
)

PHOSPHATE_SUPPLIERS = (
    ("oleyl", "fatty alcohol production, from palm oil", "fatty alcohol", "RER"),
    ("pocl3", "market for phosphorus oxychloride", "phosphorus oxychloride", "RER"),
    ("tea", "market for triethanolamine", "triethanolamine", "GLO"),
    ("hexane", "market for hexane", "hexane", "GLO"),
    ("mibk", "market for methyl isobutyl ketone", "methyl isobutyl ketone", "GLO"),
    (
        "naoh",
        "market for sodium hydroxide, without water, in 50% solution state",
        "sodium hydroxide, without water, in 50% solution state",
        "RER",
    ),
    ("di_water", "market for water, deionised", "water, deionised", "RoW"),
)


def remaining_ingredients_biosphere() -> dict[tuple[str, str], dict[str, object]]:
    """Provide synthetic emissions that distinguish manufacture from delivery."""
    result = {
        ("biosphere", code): {
            "name": name,
            "unit": "kilogram",
            "categories": ("air",),
            "type": "emission",
        }
        for code, name in (
            ("fas_marker", "Fixture whole FAS manufacture marker"),
            ("freight_marker", "Fixture freight marker"),
            ("water_marker", "Fixture tap-water supply marker"),
            ("chemical_marker", "Fixture chemical supply marker"),
        )
    }
    return result


def remaining_ingredients_native_inventory(
    database: str, *, future: bool
) -> dict[tuple[str, str], dict[str, object]]:
    """Build synthetic suppliers, an eight-supplier market and four freight modes.

    Original and future reference amounts are two and five of each stated unit.
    Distinct emissions test supplier-date selection and correct normalization.
    Fatty alcohol sulfate retains utility, waste and capital requirements.
    Package operations and treatment calculations are not replaced by fixtures.
    """
    output = 5.0 if future else 2.0
    data: dict[tuple[str, str], dict[str, object]] = {}

    def exchange(code: str, amount: float, *, biosphere: bool = False) -> dict[str, object]:
        """Attach identifiable uncertainty parameters."""
        return {
            "input": ("biosphere" if biosphere else database, code),
            "type": "biosphere" if biosphere else "technosphere",
            "amount": amount,
            "uncertainty type": 2,
            "loc": math.log(abs(amount)) + 0.02,
            "scale": 0.17,
            "minimum": amount / 2 if amount > 0 else amount * 2,
            "maximum": amount * 2 if amount > 0 else amount / 2,
            "negative": amount < 0,
            "allocation": "invented native allocation",
        }

    def process(
        code: str,
        name: str,
        product: str,
        location: str,
        rows: list[dict[str, object]],
        *,
        unit: str = "kilogram",
        production: float | None = None,
    ) -> None:
        """Store a supplier's name, product, location and synthetic reference amount."""
        data[database, code] = {
            "name": name,
            "reference product": product,
            "unit": unit,
            "location": location,
            "type": "process",
            "exchanges": [
                {
                    "input": (database, code),
                    "type": "production",
                    "amount": output if production is None else production,
                },
                *rows,
            ],
        }

    for code, name, location in FAS_SUPPLIERS:
        selected = code == "fas_petrochemical_rer"
        rows = [
            exchange(
                "fas_marker",
                output * ((0.4 if future else 0.2) if selected else 10.0),
                biosphere=True,
            )
        ]
        if selected:
            rows.extend(
                (
                    exchange("fas_utility", 0.3 * output),
                    exchange("fas_waste", -0.1 * output),
                    exchange("fas_capital", 0.02 * output),
                )
            )
        process(code, name, "fatty alcohol sulfate", location, rows)

    # Waste treatment has negative production and receives negative demand once.
    # Distinct emissions also make omitted utilities or capital inputs visible.
    for code, amount in (("fas_utility", 0.1), ("fas_waste", 0.3), ("fas_capital", 0.5)):
        process(
            code,
            f"fixture {code}",
            f"fixture {code}",
            "GLO",
            [exchange("fas_marker", amount * output, biosphere=True)],
            production=-output if code == "fas_waste" else output,
        )

    for index, product in enumerate(
        (
            "transport, freight, lorry, unspecified",
            "transport, freight train",
            "transport, freight, sea, transoceanic ship",
            "transport, freight, inland waterways, barge",
        )
    ):
        process(
            f"freight_{index}",
            f"fixture freight {index}",
            product,
            "GLO",
            [exchange("freight_marker", 0.1 * (index + 1) * output, biosphere=True)],
            unit="ton kilometer",
        )
    process(
        "fas_market",
        "market for fatty alcohol sulfate",
        "fatty alcohol sulfate",
        "GLO",
        [exchange(code, output / 8) for code, _name, _location in FAS_SUPPLIERS]
        + [
            exchange(f"freight_{index}", (index + 1) * (0.04 if future else 0.03) * output)
            for index in range(4)
        ],
    )
    market_production = cast(list[dict[str, object]], data[database, "fas_market"]["exchanges"])[0]
    market_production.update(
        {
            "activity": "invented mixed-FAS market producer",
            "flow": "invented pure mixed-FAS product",
            "name": "fatty alcohol sulfate",
            "reference product": "fatty alcohol sulfate",
            "unit": "kilogram",
            "CAS number": "invented mixed-FAS CAS",
            "chemical formula": "invented pure mixed-FAS formula",
            "formula": "native_output",
            "uncertainty type": 2,
            "loc": math.log(output),
            "scale": 0.12,
            "negative": False,
            "allocation": "invented native market allocation",
        }
    )
    if not future:
        # The generated delivery product uses its 28%-active petrochemical composition.
        market_production["properties"] = {
            "carbon content": {"amount": 0.5, "unit": "dimensionless"},
            "carbon content, fossil": {"amount": 0.25, "unit": "dimensionless"},
            "carbon content, non-fossil": {"amount": 0.25, "unit": "dimensionless"},
            "dry mass": {"amount": 1.0, "unit": "kg"},
            "water content": {"amount": 0.0, "unit": "dimensionless"},
            "heating value, net": {"amount": 16.0, "unit": "MJ"},
        }
    process(
        "tap_water",
        "market for tap water",
        "tap water",
        "Europe without Switzerland",
        [exchange("water_marker", (0.02 if future else 0.01) * output, biosphere=True)],
    )
    for code, name, product, location in PHOSPHATE_SUPPLIERS:
        process(
            code,
            name,
            product,
            location,
            [exchange("chemical_marker", (0.08 if future else 0.05) * output, biosphere=True)],
        )
    return data


@contextmanager
def remaining_phosphate_boundary(
    root: Path, *, future: bool
) -> Iterator[
    tuple[
        Any,
        PreparedProviderAuthority,
        dict[str, tuple[str, str]],
        dict[tuple[str, str], list[dict[str, object]]],
    ]
]:
    """Prepare phosphate supply and treatment from synthetic original or dated inputs.

    Every activity code and internal reference changes in the future database.
    Study treatments have a separate provider database; their quantities are
    calculated by the package's polyamide mixed-wastewater model.
    """
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    original = native_chemical_inventory(catalogue, "source")
    # Catalog recipes supply ten wastewater, sludge, and ash services, operating inputs,
    # solvent-residue handling, and ash treatment.
    service_ids = {
        "pa_tea_monooleyl_phosphate",
        "pa_ester_operation",
        "pa_phosphate_water_collection_ch",
        "pa_phosphate_water_treatment_ch",
        "pa_phosphate_sludge_incineration_market_ch",
        "pa_phosphate_sludge_incineration_ch",
        "pa_phosphate_sludge_landfarming_market_ch",
        "pa_phosphate_sludge_landfarming_ch",
        "pa_phosphate_bottom_ash_market_ch",
        "pa_phosphate_bottom_ash_treatment_ch",
        "pa_phosphate_residual_ash_market_ch",
        "pa_phosphate_residual_ash_treatment_ch",
        "pa_phosphate_solvent_residue",
        "pa_phosphate_solvent_residue_ash",
    }
    provider = replace(
        provider,
        catalogue={
            **catalogue,
            "services": [
                row
                for row in cast(list[dict[str, Any]], catalogue["services"])
                if row["provider_service_id"] in service_ids
            ],
            "routes": [],
        },
        routes=provider.routes.iloc[:0],
    )
    available = {
        (record["name"], record["reference product"], record["unit"], record["location"])
        for record in original.values()
    }
    independent = remaining_ingredients_native_inventory("source", future=False)
    for code, name, product, location in PHOSPHATE_SUPPLIERS:
        if (name, product, "kilogram", location) not in available:
            record = independent["source", code]
            # The Swiss treatment fixture already defines the fossil carbon-dioxide marker.
            for row in cast(list[dict[str, object]], record["exchanges"]):
                if row["type"] == "biosphere":
                    row["input"] = ("biosphere", "mc_co2_fossil")
            original["source", code] = record
    names = BrightwayNames(root / "brightway", "pa_phosphate_water", "biosphere", "source")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(native_chemical_biosphere(), searchable=False)
        bd.Database("source").write(original, searchable=False)
        database = "future" if future else "source"
        if future:
            keys = {key: ("future", f"future_{key[1]}") for key in original}
            dated_records: dict[tuple[str, str], dict[str, object]] = {}
            for key, record in original.items():
                dated_record = deepcopy(record)
                dated_record["database"], dated_record["code"] = keys[key]
                for row in cast(list[dict[str, Any]], dated_record["exchanges"]):
                    row["input"] = keys.get(row["input"], row["input"])
                    # Database.write adds output keys to the source dictionaries;
                    # both endpoints must follow the renamed future inventory.
                    row["output"] = keys[key]
                dated_records[keys[key]] = dated_record
            bd.Database("future").write(dated_records, searchable=False)
        before = {
            node.key: [dict(ex) for ex in node.exchanges()]
            for native_database in {"source", database}
            for node in bd.Database(native_database)
        }
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            ordinary_database=database,
            technology_database=database,
            provider=provider,
            material_recipe=physical_material_recipe(),
            resolver_cache=_NativeResolverCache("source", "biosphere"),
        )
        by_descriptor = {
            (node["name"], node["reference product"], node["unit"], node["location"]): node.key
            for node in bd.Database(database)
        }
        selected = {
            code: by_descriptor[name, product, "kilogram", location]
            for code, name, product, location in PHOSPHATE_SUPPLIERS
        }
        selected.update(
            {
                name: (frontier.database_name, provider_service_code(service_id))
                for name, service_id in (
                    ("phosphate", "pa_tea_monooleyl_phosphate"),
                    ("collection", "pa_phosphate_water_collection_ch"),
                    ("operation", "pa_ester_operation"),
                    ("solvents", "pa_phosphate_solvent_residue"),
                )
            }
        )
        yield bd, provider, selected, before
