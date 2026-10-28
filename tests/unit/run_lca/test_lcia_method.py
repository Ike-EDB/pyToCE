import json
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from pytoce.prepare_backgrounds.preparation.project_biosphere import (
    load_lcia_catalogue,
)
from pytoce.run_lca.calculation.lcia import (
    ImpactCategory,
    canonical_lcia_method,
    lcia_method_discriminators,
    normalized_lcia_discriminator,
    normalized_lcia_method,
    readable_lcia_method_selection,
    resolve_lcia_methods_from_authority,
)
from pytoce.shared.runtime.filesystem import payload_sha256


def test_lcia_method_canonicalizes_groups_and_exact_method_tuples() -> None:
    exact = '["IPCC 2021", "climate change", "GWP 100a"]'

    groups = normalized_lcia_method(("pb_lcia", "ef31", "pb_lcia"))
    assert groups.groups == ("ef31", "pb_lcia")
    assert groups.exact_method == ()
    assert canonical_lcia_method(groups) == ("ef31", "pb_lcia")
    assert lcia_method_discriminators(groups) == ("ef31", "pb_lcia")
    assert normalized_lcia_method("ef31").groups == ("ef31",)

    sequence = normalized_lcia_method(("IPCC 2021", "climate change", "GWP 100a"))
    scalar = normalized_lcia_method(exact)
    assert sequence == scalar
    assert sequence.exact_method == (
        "IPCC 2021",
        "climate change",
        "GWP 100a",
    )
    compact_exact = '["IPCC 2021","climate change","GWP 100a"]'
    assert canonical_lcia_method(sequence) == sequence.exact_method
    assert lcia_method_discriminators(sequence) == (compact_exact,)
    assert normalized_lcia_discriminator(exact) == compact_exact
    assert normalized_lcia_discriminator("ef31") == "ef31"
    assert readable_lcia_method_selection(groups) == "ef31_pb_lcia"
    assert readable_lcia_method_selection(sequence) == (
        "method__ui_up_uc_uc_s2021__climate_schange___ug_uw_up_s100a"
    )
    alias_component = normalized_lcia_method(("ef31", "custom category"))
    assert alias_component.groups == ()
    assert alias_component.exact_method == ("ef31", "custom category")

    with pytest.raises(ValueError, match="components must be non-empty strings"):
        normalized_lcia_method(("ef31", ""))
    with pytest.raises(ValueError, match="components must be non-empty strings"):
        normalized_lcia_method(cast(Any, ("ef31", 1)))

    invalid: tuple[object, ...] = (None, (), "not json", "[]", '["ok", 1]')
    for value in invalid:
        with pytest.raises(ValueError):
            normalized_lcia_method(cast(Any, value))


def test_exact_method_readable_names_are_injective_for_literal_token_words() -> None:
    literal = readable_lcia_method_selection(normalized_lcia_method(("space",)))
    symbol = readable_lcia_method_selection(normalized_lcia_method((" ",)))

    assert literal == "method_space"
    assert symbol == "method__s"
    assert literal != symbol


def test_default_alias_uses_current_catalogue_on_prepared_superset() -> None:
    specs, _ = load_lcia_catalogue()
    rows = [
        {
            "selection_order": spec.order,
            "method_id": spec.method_id,
            "method_version": spec.method_version,
            "impact_category_id": spec.impact_category_id,
            "impact_category_tuple_json": json.dumps(
                spec.impact_category_tuple,
                separators=(",", ":"),
            ),
            "unit": spec.unit,
            "factor_identity": f"factor_{spec.order}",
        }
        for spec in specs
    ]
    rows.append(
        {
            "selection_order": 27,
            "method_id": "ef31",
            "method_version": "3.1",
            "impact_category_id": "climate_change_biogenic",
            "impact_category_tuple_json": json.dumps(
                (
                    "EF v3.1",
                    "climate change: biogenic",
                    "global warming potential (GWP100)",
                ),
                separators=(",", ":"),
            ),
            "unit": "kg CO2-Eq",
            "factor_identity": "retained_superset_factor",
        }
    )

    selected = resolve_lcia_methods_from_authority(
        pd.DataFrame(rows), normalized_lcia_method("ef31"), None
    )

    assert len(selected) == 16
    assert [category.selection_order for category in selected] == list(range(1, 17))
    assert "climate_change_biogenic" not in {category.impact_category_id for category in selected}


def test_exact_registered_package_method_uses_the_prepared_factor_authority() -> None:
    ordinary = ("EF v3.1", "climate change", "global warming potential (GWP100)")
    prepared_factor = payload_sha256(
        {
            "impact_category_tuple": list(ordinary),
            "factors": [
                {
                    "database": "biosphere3",
                    "code": "carbon-dioxide",
                    "amount": 1.0,
                }
            ],
        }
    )
    live_exact_factor = payload_sha256(
        {
            "method_tuple": ordinary,
            "factors": [
                {
                    "flow_database": "biosphere3",
                    "flow_code": "carbon-dioxide",
                    "amount": 1.0,
                    "location": None,
                }
            ],
        }
    )
    assert prepared_factor != live_exact_factor
    authority = pd.DataFrame.from_records(
        [
            {
                "selection_order": 2,
                "method_id": "ef31",
                "method_version": "3.1",
                "impact_category_id": "climate_change",
                "impact_category_tuple_json": json.dumps(ordinary, separators=(",", ":")),
                "unit": "kg CO2-Eq",
                "factor_identity": prepared_factor,
            }
        ]
    )
    bd = SimpleNamespace(
        methods={ordinary: {"unit": "kg CO2-Eq"}},
        Method=lambda _method: pytest.fail("prepared exact methods must not be re-hashed"),
    )

    resolved = resolve_lcia_methods_from_authority(
        authority,
        normalized_lcia_method(ordinary),
        bd,
    )

    assert resolved == (
        ImpactCategory(
            selection_order=1,
            method_id="ef31",
            method_version="3.1",
            impact_category_id="climate_change",
            method_tuple=ordinary,
            unit="kg CO2-Eq",
            factor_identity=prepared_factor,
        ),
    )


def test_exact_method_authority_reports_missing_or_inconsistent_registration() -> None:
    method = ("EF v3.1", "climate change", "global warming potential (GWP100)")
    selection = normalized_lcia_method(method)
    row = {
        "method_id": "custom",
        "method_version": "1",
        "impact_category_id": "category",
        "impact_category_tuple_json": json.dumps(method, separators=(",", ":")),
        "unit": "kg",
        "factor_identity": "factor",
    }

    with pytest.raises(ValueError, match="not registered"):
        resolve_lcia_methods_from_authority(
            pd.DataFrame([row]),
            selection,
            SimpleNamespace(methods={}),
        )
    changed_registration = SimpleNamespace(methods={method: {"unit": "mole"}})
    with pytest.raises(ValueError, match="Brightway reports unit"):
        resolve_lcia_methods_from_authority(pd.DataFrame([row]), selection, changed_registration)
