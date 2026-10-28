"""Registered-method selection and characterization."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum
from typing import Any

import numpy as np
import pandas as pd

from pytoce.shared.runtime.filesystem import canonical_json_bytes, payload_sha256

from .timex import CharacterizationFactorBank, PrincipalInventory

_METHOD_ORDER = {"ef31": 0, "pb_lcia": 1}


@dataclass(frozen=True, slots=True)
class ImpactCategory:
    """One registered impact category selected for characterization."""

    selection_order: int
    method_id: str
    method_version: str
    impact_category_id: str
    method_tuple: tuple[str, ...]
    unit: str
    factor_identity: str


@dataclass(frozen=True, slots=True)
class CharacterizedInventory:
    """Regional total and annual scores from one principal inventory."""

    scores: pd.DataFrame
    annual_scores: pd.DataFrame


@dataclass(frozen=True, slots=True)
class LCIAMethodSelection:
    """Canonical package groups or one exact Brightway method tuple."""

    groups: tuple[str, ...]
    exact_method: tuple[str, ...]


def canonical_lcia_method(selection: LCIAMethodSelection) -> tuple[str, ...]:
    """Return the canonical public tokens for one LCIA selection."""

    return selection.groups or selection.exact_method


def readable_lcia_method_selection(selection: LCIAMethodSelection) -> str:
    """Return the collision-free readable path name for one LCIA selection."""

    if selection.groups:
        return "_".join(selection.groups)
    return "method_" + "__".join(
        _readable_method_component(value) for value in selection.exact_method
    )


def _readable_method_component(value: str) -> str:
    """Encode exact characters as compact case-insensitive identifiers.

    Single-underscore escapes preserve case and punctuation; double underscores
    separate tuple components. Literal underscores are escaped as ASCII codes.
    """
    punctuation = {
        " ": "_s",
        ".": "_d",
        ":": "_c",
        "/": "_f",
        "(": "_l",
        ")": "_r",
        "-": "_h",
    }
    tokens: list[str] = []
    for character in value:
        if "a" <= character <= "z" or "0" <= character <= "9":
            tokens.append(character)
        elif character in punctuation:
            tokens.append(punctuation[character])
        elif "A" <= character <= "Z":
            tokens.append("_u" + character.lower())
        elif character.isascii():
            tokens.append(f"_x{ord(character):02x}")
        else:
            tokens.append(f"_z{ord(character):06x}")
    return "".join(tokens)


def lcia_method_discriminators(selection: LCIAMethodSelection) -> tuple[str, ...]:
    """Return the annual-table discriminator values for a selection.

    Package groups retain their readable aliases. One exact Brightway method
    is represented by the same compact JSON-array token accepted by the scalar
    :func:`normalized_lcia_method` form.
    """

    if selection.groups:
        return selection.groups
    method = selection.exact_method
    return (json.dumps(method, ensure_ascii=True, separators=(",", ":")),)


def normalized_lcia_discriminator(value: str) -> str:
    """Validate and canonicalize one annual-table LCIA discriminator."""

    if value in _METHOD_ORDER:
        return value
    selection = normalized_lcia_method(value)
    return lcia_method_discriminators(selection)[0]


def normalized_lcia_method(
    lcia_method: str | Sequence[str],
) -> LCIAMethodSelection:
    """Return canonical groups or one exact Brightway method tuple.

    A scalar exact selection uses a JSON array string. A non-alias sequence is
    treated directly as one complete Brightway method tuple.
    """

    if isinstance(lcia_method, str):
        if lcia_method in _METHOD_ORDER:
            return LCIAMethodSelection(groups=(lcia_method,), exact_method=())
        try:
            decoded = json.loads(lcia_method)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "An exact Brightway method must be a JSON array of method-tuple strings"
            ) from exc
        if (
            not isinstance(decoded, list)
            or not decoded
            or any(not isinstance(item, str) or not item for item in decoded)
        ):
            raise ValueError(
                "An exact Brightway method must be a JSON array of method-tuple strings"
            )
        return LCIAMethodSelection(groups=(), exact_method=tuple(decoded))

    try:
        raw = tuple(lcia_method)
    except TypeError as exc:
        raise ValueError(
            "lcia_method must be one method name or a sequence of method-tuple strings"
        ) from exc
    if not raw:
        raise ValueError("lcia_method cannot be empty")
    if any(not isinstance(value, str) or not value for value in raw):
        raise ValueError("Brightway method tuple components must be non-empty strings")
    aliases = tuple(value in _METHOD_ORDER for value in raw)
    if all(aliases):
        group_set = set(raw)
        return LCIAMethodSelection(
            groups=tuple(sorted(group_set, key=_METHOD_ORDER.__getitem__)),
            exact_method=(),
        )
    return LCIAMethodSelection(
        groups=(),
        exact_method=raw,
    )


def _default_categories_from_authority(
    table: pd.DataFrame,
    selections: tuple[str, ...],
) -> tuple[ImpactCategory, ...]:
    """Read the selected package LCIA categories from the prepared method table.

    The installed LCIA catalog supplies the required method identifiers,
    versions, category tuples, native units, and display order. The prepared
    table contains the selected EF v3.1 and PB-LCIA rows. Returned categories
    preserve catalog order and their recorded factor identities using their recorded factor
    identities.
    """
    from pytoce.prepare_backgrounds.preparation.project_biosphere import (
        load_lcia_catalogue,
    )

    specs, _ = load_lcia_catalogue()
    authority = {
        (
            spec.method_id,
            spec.method_version,
            spec.impact_category_id,
            spec.impact_category_tuple,
            spec.unit,
        ): spec.order
        for spec in specs
        if spec.method_id in selections
    }
    selected_rows: list[tuple[int, Any]] = []
    for row in table.itertuples(index=False):
        coordinate = (
            str(row.method_id),
            str(row.method_version),
            str(row.impact_category_id),
            tuple(json.loads(str(row.impact_category_tuple_json))),
            str(row.unit),
        )
        selection_order = authority.get(coordinate)
        if selection_order is None:
            continue
        selected_rows.append((selection_order, row))
    selected_rows.sort(key=lambda item: item[0])
    return tuple(
        ImpactCategory(
            selection_order=selection_order,
            method_id=str(row.method_id),
            method_version=str(row.method_version),
            impact_category_id=str(row.impact_category_id),
            method_tuple=tuple(json.loads(str(row.impact_category_tuple_json))),
            unit=str(row.unit),
            factor_identity=str(row.factor_identity),
        )
        for selection_order, row in selected_rows
    )


def resolve_lcia_methods_from_authority(
    authority: pd.DataFrame,
    lcia_method: LCIAMethodSelection,
    bd: Any,
) -> tuple[ImpactCategory, ...]:
    """Resolve methods from the prepared LCIA table and actual Brightway registration.

    Package method groups are read from the supplied table. An
    exact conventional method that is present in that table uses the table's
    recorded labels, unit, and factor identity after its Brightway registration
    is confirmed. Other exact methods use the ordinary Brightway resolution
    path. Their identity preserves the native unit, central amounts, and stored
    uncertainty parameters using their registered central values.
    """

    if lcia_method.groups:
        return _default_categories_from_authority(authority, lcia_method.groups)

    method = lcia_method.exact_method
    if method not in bd.methods:
        raise ValueError(f"Brightway method is not registered: {method!r}")
    metadata = bd.methods[method]
    discriminator = json.dumps(method, ensure_ascii=True, separators=(",", ":"))
    prepared = authority.loc[authority["impact_category_tuple_json"].astype(str).eq(discriminator)]
    if not prepared.empty:
        row = prepared.iloc[0]
        unit = str(metadata.get("unit", ""))
        if str(row["unit"]) != unit:
            raise ValueError(
                f"Brightway reports unit {unit!r} for method {method!r}, but the prepared "
                f"LCIA table records {str(row['unit'])!r}. Create a new workspace and run "
                "prepare_backgrounds() again."
            )
        factor_identity = str(row["factor_identity"])
        return (
            ImpactCategory(
                selection_order=1,
                method_id=str(row["method_id"]),
                method_version=str(row["method_version"]),
                impact_category_id=str(row["impact_category_id"]),
                method_tuple=method,
                unit=unit,
                factor_identity=factor_identity,
            ),
        )
    factors: list[dict[str, Any]] = []
    for line in bd.Method(method).load():
        flow = bd.get_node(id=int(line[0]))
        native = line[1]
        factor = {
            "flow_database": str(flow.key[0]),
            "flow_code": str(flow.key[1]),
            "amount": float(native["amount"] if isinstance(native, dict) else native),
            "location": None if len(line) == 2 else str(line[2]),
        }
        if isinstance(native, dict):
            factor["uncertainty"] = {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in native.items()
                if key != "amount"
            }
        factors.append(factor)
    factors.sort(
        key=lambda row: (
            row["flow_database"],
            row["flow_code"],
            "" if row["location"] is None else str(row["location"]),
            row["amount"],
            canonical_json_bytes(row["uncertainty"]) if "uncertainty" in row else b"",
        )
    )
    unit = str(metadata.get("unit", ""))
    return (
        ImpactCategory(
            selection_order=1,
            method_id=method[0],
            method_version="",
            impact_category_id=method[-1],
            method_tuple=method,
            unit=unit,
            factor_identity=payload_sha256(
                {
                    "method_tuple": method,
                    "unit": unit,
                    "factors": factors,
                }
            ),
        ),
    )


def _annual_inventory(
    principal: PrincipalInventory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect annual regional inventory amounts in canonical coordinate order."""
    inventory = principal.inventory
    years = inventory["year"].to_numpy(dtype=np.int16, copy=False)
    flow_rows = np.fromiter(
        (int(flow_id) for flow_id in inventory["flow_id"]),
        dtype=np.int64,
        count=len(inventory),
    )
    amounts = inventory["amount"].to_numpy(dtype=np.float64, copy=False)
    order = np.lexsort((flow_rows, years))
    return (
        years[order],
        flow_rows[order],
        amounts[order],
    )


def characterize_principal_inventory(
    principal: PrincipalInventory,
    methods: tuple[ImpactCategory, ...],
    factor_bank: CharacterizationFactorBank,
    *,
    foreground_scenario: str,
    background_scenario: str,
    background_dimensions: dict[str, object],
    region: str,
) -> CharacterizedInventory:
    """Characterize one regional inventory with the selected LCIA methods.

    ``principal`` contains biosphere amounts indexed by activity year and Brightway flow id. For
    each method, every amount is multiplied by
    its registered characterization factor and stably summed by year and across
    2019--2100. Returned annual and total scores include the method's native unit
    and the foreground, background, and region coordinates.
    """

    years, flow_ids, amounts = _annual_inventory(principal)
    bounds = {
        year: (
            int(np.searchsorted(years, year, side="left")),
            int(np.searchsorted(years, year, side="right")),
        )
        for year in range(2019, 2101)
    }

    score_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []
    for method in methods:
        factors = factor_bank.factors[method.method_tuple]
        characterized = amounts * np.fromiter(
            (factors.get(int(flow_id), 0.0) for flow_id in flow_ids),
            dtype=np.float64,
            count=len(flow_ids),
        )
        annual_values = {
            year: fsum(float(value) for value in characterized[start:stop])
            for year, (start, stop) in bounds.items()
        }
        score = fsum(float(value) for value in characterized)
        dimensions = {
            "foreground_scenario": foreground_scenario,
            "background_scenario": background_scenario,
            **background_dimensions,
            "region": region,
            "method_id": method.method_id,
            "method_version": method.method_version,
            "impact_category_id": method.impact_category_id,
            "impact_category_tuple_json": json.dumps(
                method.method_tuple,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            "unit": method.unit,
        }
        score_rows.append({**dimensions, "score": score})
        annual_rows.extend(
            {**dimensions, "year": year, "score": annual_values[year]} for year in range(2019, 2101)
        )
    return CharacterizedInventory(
        scores=pd.DataFrame(score_rows),
        annual_scores=pd.DataFrame(annual_rows),
    )
