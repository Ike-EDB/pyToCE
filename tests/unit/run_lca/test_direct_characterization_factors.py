"""Brightway processing of synthetic factors for direct elementary flows."""

from dataclasses import dataclass
from types import ModuleType

import bw2calc as bc
import numpy as np
import pytest

from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    begin_matrix_contributions,
    create_provider_response_state,
)
from pytoce.run_lca.calculation.direct_contributions import (
    admit_direct_characterization_factors,
)


@dataclass(frozen=True)
class _FactorBoundary:
    """Synthetic methods and separate supplier and reference inventories in Brightway."""

    bd: ModuleType
    ids: dict[str, int]
    methods: tuple[ContributionMethod, ...]
    supplier: bc.LCA


@pytest.fixture
def factor_boundary(contribution_brightway: ModuleType) -> _FactorBoundary:
    """Register synthetic flows, inventories and methods with Brightway.

    The sparse supplier emits one flow. A separate reference activity emits every
    flow so ordinary Brightway characterization supplies an independent comparison
    for the processed method.
    """
    bd = contribution_brightway
    codes = ("supplier", "precise", "duplicate", "zero", "absent", "regional", "global")
    bd.Database("biosphere").write(
        {
            ("biosphere", code): {
                "name": code,
                "unit": "kilogram",
                "categories": ("air", "urban air close to ground"),
                "type": "emission",
            }
            for code in codes
        }
    )
    for database, emitted in (("supplier", ("supplier",)), ("reference", codes)):
        bd.Database(database).write(
            {
                (database, "process"): {
                    "name": database,
                    "reference product": "product",
                    "unit": "kilogram",
                    "location": "GLO",
                    "type": "process",
                    "exchanges": [
                        {
                            "input": (database, "process"),
                            "amount": 1.0,
                            "type": "production",
                        },
                        *[
                            {
                                "input": ("biosphere", code),
                                "amount": 1.0,
                                "type": "biosphere",
                            }
                            for code in emitted
                        ],
                    ],
                }
            }
        )
    first = ("validation", "first")
    second = ("validation", "second")
    for method in (first, second):
        bd.Method(method).register(unit="validation impact/kg")
    bd.Method(first).write(
        [
            (("biosphere", "supplier"), 2.0),
            # Static LCIA uses exchange amount.
            (("biosphere", "precise"), {"amount": 0.1, "uncertainty type": 2, "loc": 3.0}),
            (("biosphere", "duplicate"), 0.1),
            (("biosphere", "duplicate"), 0.2),
            (("biosphere", "zero"), 0.0),
            (("biosphere", "regional"), 7.0, "DE"),
            (("biosphere", "global"), 5.0, bd.config.global_location),
            (("biosphere", "global"), 17.0, "DE"),
        ]
    )
    bd.Method(second).write(
        [
            (("biosphere", "supplier"), 3.0),
            (("biosphere", "precise"), -0.25),
            (("biosphere", "zero"), 0.0),
        ]
    )
    methods = tuple(
        ContributionMethod(
            method_id="validation",
            method_version="1",
            impact_category_id=method[-1],
            method_tuple=method,
            unit="validation impact/kg",
        )
        for method in (second, first)
    )
    supplier = bc.LCA({bd.get_node(key=("supplier", "process")): 1.0}, method=first)
    supplier.lci()
    return _FactorBoundary(
        bd,
        {code: int(bd.get_node(key=("biosphere", code)).id) for code in codes},
        methods,
        supplier,
    )


def test_direct_factors_preserve_native_precision_presence_and_method_order(
    factor_boundary: _FactorBoundary,
) -> None:
    """Match factors for flows absent from the supplier to a separate Brightway inventory."""
    boundary = factor_boundary
    codes = ("global", "zero", "precise", "absent", "regional", "duplicate")
    requested = tuple(boundary.ids[code] for code in codes)
    assert not set(requested).intersection(boundary.supplier.dicts.biosphere)
    state = create_provider_response_state(boundary.bd, boundary.methods)
    result = admit_direct_characterization_factors(state, requested)

    np.testing.assert_array_equal(result.flow_ids, requested)
    np.testing.assert_array_equal(
        result.present,
        [[False, True, True, False, False, False], [True, True, True, False, False, True]],
    )
    expected = np.array(
        [
            [0.0, 0.0, -0.25, 0.0, 0.0, 0.0],
            [5.0, 0.0, float(np.float32(0.1)), 0.0, 0.0, float(np.float32(0.3))],
        ]
    )
    np.testing.assert_array_equal(result.factors, expected)
    assert result.factors[1, 2] != 0.1
    # Duplicate processing sums stored float32 values before the float64 matrix.
    assert result.factors[1, 5] != float(np.float32(0.1)) + float(np.float32(0.2))
    reference = bc.LCA({boundary.bd.get_node(key=("reference", "process")): 1.0})
    reference.lci()
    for index, method in enumerate(boundary.methods):
        # bw2calc 2.5.0 uses a typing.Union object as the default instead of an
        # annotation; its documented public method argument is a method tuple.
        reference.switch_method(method.method_tuple)  # pyright: ignore[reportArgumentType]
        vector = np.asarray(reference.characterization_matrix.sum(axis=0)).ravel()
        np.testing.assert_array_equal(
            result.factors[index], [vector[reference.dicts.biosphere[flow]] for flow in requested]
        )
        admitted = state.characterization_factors[method.method_tuple]
        assert admitted[boundary.ids["zero"]] == 0.0
        assert boundary.ids["absent"] not in admitted
        assert boundary.ids["regional"] not in admitted

    # Later supplier method switching/cache reuse must retain direct-only factors.
    begin_matrix_contributions(state, boundary.supplier, "supplier")
    begin_matrix_contributions(state, boundary.supplier, "supplier")
    for index, method in enumerate(boundary.methods):
        admitted = state.characterization_factors[method.method_tuple]
        assert admitted[boundary.ids["zero"]] == 0.0
        assert admitted[boundary.ids["precise"]] == expected[index, 2]
        assert admitted[boundary.ids["supplier"]] == (3.0, 2.0)[index]


def test_required_flows_with_no_applicable_method_rows_keep_explicit_absence(
    factor_boundary: _FactorBoundary,
) -> None:
    """Keep absent conventional LCIA factors distinct from zero-valued contributions."""
    boundary = factor_boundary
    state = create_provider_response_state(boundary.bd, boundary.methods)
    requested = (boundary.ids["regional"], boundary.ids["absent"])
    result = admit_direct_characterization_factors(state, requested)
    np.testing.assert_array_equal(result.factors, np.zeros((2, 2)))
    np.testing.assert_array_equal(result.present, np.zeros((2, 2), dtype=bool))
    assert state.characterization_factors == {
        method.method_tuple: {} for method in boundary.methods
    }


def test_nonfinite_processed_direct_factor_is_rejected_outside_supplier_biosphere(
    factor_boundary: _FactorBoundary,
) -> None:
    """Finite source input that overflows native processing fails at direct admission."""
    boundary = factor_boundary
    method = boundary.methods[0]
    # The finite characterization input exceeds the processed float32 range. Direct-flow
    # characterization checks this independently of supplier emissions.
    with pytest.warns(RuntimeWarning, match="overflow"):
        boundary.bd.Method(method.method_tuple).write(
            [(boundary.ids["supplier"], 3.0), (boundary.ids["precise"], 1e40)]
        )
    state = create_provider_response_state(boundary.bd, (method,))
    with pytest.raises(ValueError, match="nonfinite processed characterization factors"):
        admit_direct_characterization_factors(state, (boundary.ids["precise"],))
