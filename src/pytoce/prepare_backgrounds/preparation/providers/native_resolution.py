"""Deterministic resolution of native activities, flows, and exchanges."""

from typing import Any, cast


class _ActivityResolver:
    """Resolve accepted semantic activity keys in one actual database."""

    def __init__(self, bd: Any, database_name: str) -> None:
        """Index activities by Brightway key and by name, product, unit and location."""
        self.database_name = database_name
        activities: tuple[Any, ...] = tuple(bd.Database(database_name))
        self._by_key = {cast(tuple[str, str], activity.key): activity for activity in activities}
        self._by_descriptor: dict[tuple[str, str, str, str], Any] = {}
        for activity in activities:
            key = (
                str(activity.get("name", "")),
                str(activity.get("reference product", "")),
                str(activity.get("unit", "")),
            )
            self._by_descriptor[(*key, str(activity.get("location", "")))] = activity

    def node(self, key: tuple[str, str]) -> Any:
        """Return a node from the index already built for this database."""

        return self._by_key[key]

    def activity(self, selector: dict[str, object], *, year: int) -> Any:
        """Return the declared dataset in the catalog's anchor period.

        Native ordinary suppliers use ecoinvent through 2025 and matching-member
        premise from 2030. Where their locations differ, the catalog declares
        both choices. Premise-only technology selectors retain their own fixed
        location, including at the 2020 and 2025 technology anchors.
        """

        if "location_by_period" in selector:
            locations = cast(dict[str, str], selector["location_by_period"])
            location = locations["baseline" if year < 2030 else "future"]
        else:
            location = str(selector["exact_location"])
        return self._by_descriptor[
            (
                str(selector["name"]),
                str(selector["reference_product"]),
                str(selector["unit"]),
                location,
            )
        ]


class _FlowResolver:
    """Resolve exact biosphere identities from the selected source."""

    def __init__(self, bd: Any, database_name: str) -> None:
        """Index biosphere flows by Brightway key and full semantic identity."""
        self.database_name = database_name
        self._flows: dict[tuple[str, str, tuple[str, ...]], Any] = {}
        self._by_key: dict[tuple[str, str], Any] = {}
        for flow in bd.Database(database_name):
            key = (
                str(flow.get("name", "")),
                str(flow.get("unit", "")),
                tuple(str(value) for value in flow.get("categories", ())),
            )
            self._flows[key] = flow
            self._by_key[cast(tuple[str, str], flow.key)] = flow

    def node(self, key: tuple[str, str]) -> Any:
        """Return a flow from the index already built for this database."""

        return self._by_key[key]

    def flow(self, selector: dict[str, object]) -> Any:
        """Return one exact biosphere flow."""

        return self._flows[
            (
                str(selector["name"]),
                str(selector["unit"]),
                tuple(str(value) for value in cast(list[object], selector["categories"])),
            )
        ]


class _NativeResolverCache:
    """Reuse shared indexes and the current member anchor during preparation.

    Provider compilation and PA source admission read the same immutable native
    anchor. Keep the most recent index for reuse within the current member and year.
    """

    def __init__(self, source_database: str, biosphere_database: str) -> None:
        """Initialize shared and current-anchor indexes lazily."""
        self.source_database = source_database
        self.biosphere_database = biosphere_database
        self._source: _ActivityResolver | None = None
        self._biosphere: _FlowResolver | None = None
        self._dated: _ActivityResolver | None = None

    def activities(self, bd: Any, database_name: str) -> _ActivityResolver:
        """Return the one semantic activity index for a prepared database."""

        if database_name != self.source_database:
            if self._dated is None or self._dated.database_name != database_name:
                self._dated = _ActivityResolver(bd, database_name)
            return self._dated
        if self._source is None:
            self._source = _ActivityResolver(bd, database_name)
        return self._source

    def flows(self, bd: Any) -> _FlowResolver:
        """Return the one exact flow index for a prepared biosphere database."""

        if self._biosphere is None:
            self._biosphere = _FlowResolver(bd, self.biosphere_database)
        return self._biosphere


def _input_semantics(node: Any) -> tuple[str, str, str, str]:
    """Describe the exchange fields that determine its recipe meaning."""
    return (
        str(node.get("name", "")),
        str(node.get("reference product", "")),
        str(node.get("unit", "")),
        str(node.get("location", "")),
    )


def _exchange_record(exchange: Any, node: Any) -> dict[str, object]:
    """Project one exchange's supplier identity and amount into a provenance row."""
    return {
        "database_name": str(node.key[0]),
        "code": str(node.key[1]),
        "name": str(node.get("name", "")),
        "reference_product": str(node.get("reference product", "")),
        "unit": str(node.get("unit", "")),
        "location": str(node.get("location", "")),
        "categories": [str(value) for value in node.get("categories", ())],
        "exchange_type": str(exchange["type"]),
        "amount": float(exchange["amount"]),
    }


def _exchange_semantic_key(
    exchange_type: str,
    node: Any,
) -> tuple[str, str, str, str, tuple[str, ...], str]:
    """Return the allocation-independent semantic identity of one exchange."""

    return (
        str(node.get("name", "")),
        str(node.get("reference product", "")),
        str(node.get("unit", "")),
        str(node.get("location", "")),
        tuple(str(value) for value in node.get("categories", ())),
        exchange_type,
    )


def _matches(node: Any, selector: dict[str, object]) -> bool:
    """Match declared exchange names, product, unit and any specified location.

    A name list selects all matching inputs together, for example electricity
    market names that premise merges. Exact provider-dataset resolution has a separate owner.
    A product-and-unit declaration without a name selects every dated supplier
    of that feed, as required for the PDO phosphate-salt main reactants.
    """
    name = str(node.get("name", ""))
    expected = selector.get("name")
    name_matches = expected is None or (
        name in cast(list[str], expected) if isinstance(expected, list) else name == str(expected)
    )
    return (
        name_matches
        and str(node.get("reference product", "")) == str(selector["reference_product"])
        and str(node.get("unit", "")) == str(selector["unit"])
        and (
            "location" not in selector or str(node.get("location", "")) == str(selector["location"])
        )
    )


def _add_amount(
    exchanges: dict[tuple[tuple[str, str], str], float],
    key: tuple[str, str],
    exchange_type: str,
    amount: float,
) -> None:
    """Accumulate an amount by supplier and exchange type in encounter order."""
    group = (key, exchange_type)
    exchanges[group] = exchanges.get(group, 0.0) + amount


def _exact_exchange_matches(
    exchange: Any,
    input_node: Any,
    selector: dict[str, object],
) -> bool:
    """Select exchanges that match every field of an exact recipe selector."""
    amount = float(exchange["amount"])
    sign = str(selector["sign"])
    return (
        str(exchange["type"]) == str(selector["exchange_type"])
        and str(input_node.get("name", "")) == str(selector["name"])
        and str(input_node.get("reference product", "")) == str(selector["reference_product"])
        and str(input_node.get("unit", "")) == str(selector["unit"])
        and str(input_node.get("location", "")) == str(selector["location"])
        and ((amount < 0.0) if sign == "negative" else (amount > 0.0))
    )
