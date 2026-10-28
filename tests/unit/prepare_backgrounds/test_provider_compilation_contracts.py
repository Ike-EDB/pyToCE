"""Provider selector, production, and treatment failure contracts."""

from collections.abc import Iterable

import pytest
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    chemical_recycling,
    native_recipe,
    native_resolution,
    recipe_compilation,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import (
    _RecipeContext,
)


class _Node(dict[str, object]):
    """Brightway node double exposing its documented mapping and key surfaces."""

    def __init__(
        self,
        database: str,
        code: str,
        *,
        name: str,
        product: str,
        unit: str = "kilogram",
        location: str = "GLO",
    ) -> None:
        super().__init__(
            database=database,
            code=code,
            name=name,
            **{"reference product": product},
            unit=unit,
            location=location,
        )
        self.key = (database, code)
        self._exchanges: tuple[_Exchange, ...] = ()

    def exchanges(self) -> Iterable["_Exchange"]:
        """Return the node's public exchange iterator."""
        return iter(self._exchanges)


class _Exchange(dict[str, object]):
    """Brightway exchange double retaining the resolved input node."""

    def __init__(self, input_node: _Node, *, amount: float, exchange_type: str) -> None:
        super().__init__(input=input_node.key, amount=amount, type=exchange_type)
        self.input = input_node


class _Brightway:
    """Brightway data boundary double returning database node iterables."""

    def __init__(self, databases: dict[str, tuple[_Node, ...]]) -> None:
        self.databases = databases

    def Database(self, name: str) -> tuple[_Node, ...]:  # noqa: N802
        """Return one exact database iterable."""
        return self.databases[name]


def _selector(
    *,
    name: str = "conversion operation",
    product: str = "converted plastic",
    location: str = "GLO",
) -> dict[str, object]:
    """Return an exact native activity selector."""
    return {
        "name": name,
        "reference_product": product,
        "unit": "kilogram",
        "exact_location": location,
    }


def _semantic_selector(name: str, product: str) -> dict[str, object]:
    """Return the name/product/unit portion shared by exchange selectors."""
    return {
        "name": name,
        "reference_product": product,
        "unit": "kilogram",
    }


def _context(nodes: tuple[_Node, ...]) -> _RecipeContext:
    """Build the real recipe context over bounded Brightway node doubles."""
    bd = _Brightway({"ordinary": nodes, "biosphere": ()})
    activities = native_resolution._ActivityResolver(bd, "ordinary")
    biosphere = native_resolution._FlowResolver(bd, "biosphere")
    catalogue = provider_catalogue.load_provider_catalogue()
    return _RecipeContext(
        background_scenario="remind_ssp2_npi",
        year=2030,
        provider_database="provider_frontier",
        source=activities,
        ordinary=activities,
        technology=activities,
        biosphere=biosphere,
        material_recipe=physical_material_recipe(),
        catalogue=catalogue,
        method_authorities={
            str(row["method_authority_id"]): row
            for row in provider_catalogue._method_authority_rows(catalogue)
        },
        complete_source_ids=provider_catalogue._service_source_closures(catalogue),
        semantic_rows=[],
        technology_rows=[],
        traced_services=set(),
    )


def test_native_resolver_cache_reuses_shared_and_current_anchor_without_member_borrowing() -> None:
    """Reuse an immutable anchor, then resolve the same product in the next member."""
    source = _Node("source", "source", name="source", product="source")
    member = _Node("member", "member", name="supply", product="material")
    next_member = _Node("next_member", "other", name="supply", product="material")
    flow = _Node("biosphere", "flow", name="flow", product="")
    bd = _Brightway(
        {
            "source": (source,),
            "member": (member,),
            "next_member": (next_member,),
            "biosphere": (flow,),
        }
    )
    cache = native_resolution._NativeResolverCache("source", "biosphere")

    source_resolver = cache.activities(bd, "source")
    assert cache.activities(bd, "source") is source_resolver
    member_resolver = cache.activities(bd, "member")
    assert member_resolver is not source_resolver
    assert cache.activities(bd, "member") is member_resolver
    selector = _selector(name="supply", product="material")
    assert member_resolver.activity(selector, year=2050).key == ("member", "member")
    next_resolver = cache.activities(bd, "next_member")
    assert next_resolver.activity(selector, year=2050).key == ("next_member", "other")
    # Returning to an earlier member rebuilds that index: the request retains
    # only the current dated database, rather than all member/anchor inventories.
    assert cache.activities(bd, "member") is not member_resolver
    assert cache.activities(bd, "source") is source_resolver
    flow_resolver = cache.flows(bd)
    assert cache.flows(bd) is flow_resolver


def test_native_operation_rejects_coproduct_or_nonself_production() -> None:
    """Require the one-product allocation boundary used by extracted operations."""
    operation = _Node(
        "ordinary",
        "operation",
        name="operation",
        product="product",
    )
    coproduct = _Node(
        "ordinary",
        "coproduct",
        name="coproduct",
        product="coproduct",
    )
    exchanges = (
        _Exchange(operation, amount=1.0, exchange_type="production"),
        _Exchange(coproduct, amount=0.2, exchange_type="production"),
    )

    with pytest.raises(ValueError, match="one self-production exchange and no coproduct"):
        native_recipe._self_production_amount(operation, exchanges, method="bounded operation")


@pytest.mark.parametrize(
    ("include_production", "include_processed", "production_amount", "message"),
    [
        (False, True, 1.0, "one self-production exchange"),
        (True, False, 1.0, "one source-defined processed output"),
        (True, True, 0.0, "requires a positive mass basis"),
    ],
)
def test_chemical_preparation_requires_exact_tracked_plastic_mass_basis(
    include_production: bool,
    include_processed: bool,
    production_amount: float,
    message: str,
) -> None:
    """Reject preparation inventories missing an exact output class or positive basis."""
    operation = _Node(
        "ordinary",
        "preparation",
        name="preparation",
        product="prepared waste",
    )
    processed = _Node(
        "ordinary",
        "processed",
        name="processed plastic",
        product="processed plastic",
    )
    reject = _Node(
        "ordinary",
        "reject",
        name="combustible reject",
        product="combustible reject",
    )
    exchanges: list[_Exchange] = []
    if include_production:
        exchanges.append(_Exchange(operation, amount=production_amount, exchange_type="production"))
    if include_processed:
        exchanges.append(_Exchange(processed, amount=-0.8, exchange_type="technosphere"))
    exchanges.append(_Exchange(reject, amount=-0.2, exchange_type="technosphere"))
    operation._exchanges = tuple(exchanges)
    context = _context((operation, processed, reject))
    construction = {
        "source_role": "native_ecoinvent_provider",
        "selector": _selector(name="preparation", product="prepared waste"),
        "processed_output_selector": {
            "exchange_type": "technosphere",
            "sign": "negative",
            "name": "processed plastic",
            "reference_product": "processed plastic",
            "unit": "kilogram",
            "location": "GLO",
        },
        "plastic_reject_output_selector": {
            "exchange_type": "technosphere",
            "sign": "negative",
            "name": "combustible reject",
            "reference_product": "combustible reject",
            "unit": "kilogram",
            "location": "GLO",
        },
    }

    with pytest.raises(ValueError, match=message):
        chemical_recycling._chemical_preparation_details(context, construction)


@pytest.mark.parametrize(
    ("retained_count", "message"),
    [
        (0, "one production and one processed-feed denominator"),
        (2, "repeats a retained semantic exchange"),
    ],
)
def test_hydrothermal_sibling_requires_one_denominator_and_unique_semantic_vector(
    retained_count: int,
    message: str,
) -> None:
    """Reject an unnormalizable or semantically duplicated native conversion sibling."""
    operation = _Node(
        "ordinary",
        "hydrothermal",
        name="hydrothermal conversion",
        product="naphtha",
    )
    feed = _Node(
        "ordinary",
        "feed",
        name="processed plastic feed",
        product="processed plastic feed",
    )
    auxiliary = _Node(
        "ordinary",
        "auxiliary",
        name="process energy",
        product="process energy",
    )
    exchanges = [_Exchange(operation, amount=1.0, exchange_type="production")]
    if retained_count:
        exchanges.append(_Exchange(feed, amount=1.0, exchange_type="technosphere"))
        exchanges.extend(
            _Exchange(auxiliary, amount=0.1, exchange_type="technosphere")
            for _index in range(retained_count)
        )
    operation._exchanges = tuple(exchanges)
    context = _context((operation, feed, auxiliary))
    denominator_selector = {
        "exchange_type": "technosphere",
        "sign": "positive",
        "name": "processed plastic feed",
        "reference_product": "processed plastic feed",
        "unit": "kilogram",
        "location": "GLO",
    }

    with pytest.raises(ValueError, match=message):
        chemical_recycling._conversion_sibling_vector(
            context,
            operation,
            denominator_selector,
        )


@pytest.mark.parametrize(
    ("denominator_amounts", "message"),
    [
        ((), "normalization input is absent or nonpositive"),
        ((1.0,), "declared omitted technosphere exchange is absent"),
        ((1e308, 1e308), "sum is nonfinite"),
    ],
)
def test_input_normalized_operation_requires_denominator_and_declared_omissions(
    denominator_amounts: tuple[float, ...],
    message: str,
) -> None:
    """Reject native operations whose declared normalization boundary is incomplete."""
    operation = _Node(
        "ordinary",
        "operation",
        name="recycling operation",
        product="recycling service",
    )
    denominator = _Node(
        "ordinary",
        "denominator",
        name="incoming waste plastic",
        product="waste plastic",
    )
    exchanges = [_Exchange(operation, amount=1.0, exchange_type="production")]
    exchanges.extend(
        _Exchange(denominator, amount=amount, exchange_type="technosphere")
        for amount in denominator_amounts
    )
    operation._exchanges = tuple(exchanges)
    context = _context((operation, denominator))
    construction = {
        "kind": "input_normalized_operation",
        "base": {
            "source_role": "native_ecoinvent_provider",
            "selector": _selector(
                name="recycling operation",
                product="recycling service",
            ),
        },
        "denominator_selector": _semantic_selector("incoming waste plastic", "waste plastic"),
        "omit_technosphere": [
            _semantic_selector("separately owned material", "incorporated material")
        ],
    }

    service = next(
        service
        for service in context.catalogue["services"]
        if service["provider_service_id"] == "mechanical_recycling_hdpe"
    )
    with pytest.raises(ValueError, match=message):
        recipe_compilation._service_dataset(
            context, {**service, "construction": construction}, {}, "provider_frontier"
        )


def test_route_contexts_visit_shared_nested_service_once_per_foreground_route() -> None:
    """Trace a diamond-shaped dependency with shared ancestry visited once."""
    services = {
        "root": {
            "construction": {
                "left": {"service_id": "left"},
                "right": {"service_id": "right"},
            }
        },
        "left": {"construction": {"shared": {"service_id": "shared"}}},
        "right": {"construction": {"shared": {"service_id": "shared"}}},
        "shared": {"construction": {}},
    }
    route = {"route_id": "bounded", "provider_service_id": "root"}

    contexts = recipe_compilation._route_contexts([route], services)

    assert set(contexts) == {"root", "left", "right", "shared"}
    assert all(rows == [route] for rows in contexts.values())


def test_recipe_context_resolves_foreign_exchange_input_from_external_node() -> None:
    """Retain a technosphere input outside the two indexed provider databases."""
    operation = _Node("ordinary", "operation", name="operation", product="product")
    foreign = _Node("foreign", "input", name="foreign input", product="input")
    exchange = _Exchange(foreign, amount=1.0, exchange_type="technosphere")
    context = _context((operation,))

    assert context.input(exchange) is foreign
    assert context.input(exchange) is foreign
