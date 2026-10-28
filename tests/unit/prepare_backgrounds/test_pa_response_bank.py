"""Native PA responses use the real prepared graph and portable product identities."""

from dataclasses import replace
from pathlib import Path

import bw2calc as bc  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.providers.catalogue import provider_service_code
from pytoce.run_lca.calculation.lcia import ImpactCategory
from pytoce.run_lca.calculation.pa_receiving_endpoints import pa_receiving_endpoints
from pytoce.run_lca.calculation.providers import PreparedProviderProduct
from pytoce.run_optimization.calculation.response_kernel import (
    _direct_factor_table,
    _member_response,
    _method_registry,
)


@pytest.mark.parametrize("original", (True, False))
def test_actual_native_response_and_direct_factors_preserve_source_context(
    tmp_path: Path, original: bool
) -> None:
    """Characterize complete native suppliers and keep explicit-zero/absent CFs.

    The source portfolio has native signed waste inputs, electricity, heat,
    transport and complete feedback functions. Independent Brightway LCA checks
    one actual complete PAM endpoint; no unit-load receiving simulation defines
    the required endpoint axis. Reversed method-name ordering protects the
    distinction between registered method order and stored registry sorting.
    """
    root = tmp_path / "native"
    names = BrightwayNames(root / "brightway", "pa_receiving", "biosphere", "original")
    with pa_event_receiving_boundary(root, original=original) as boundary:
        source = boundary.sources
        endpoints, flows = pa_receiving_endpoints(boundary.reference, source)
        method_tuples = (("PA response", "z"), ("PA response", "a"))
        factors = (1.234567891, -0.345678912)
        methods = tuple(
            ImpactCategory(
                index, method[-1], "fixture", method[-1], method, "kg eq", str(index) * 64
            )
            for index, method in enumerate(method_tuples)
        )
        for method, factor in zip(method_tuples, factors, strict=True):
            native_method = boundary.bd.Method(method)
            native_method.register(unit="kg eq")
            native_method.write(
                [(flow.key, 0.0 if index == 1 else factor) for index, flow in enumerate(flows[:-1])]
            )
        registry = _method_registry(methods, ("z", "a"))
        providers = PreparedProviderProduct(
            {source.provider_anchor_year: "frontier"},
            {},
            boundary.reference,
            {
                # The ignored 2020 calibration slot is deliberately distinct
                # from the actual source-year selection; it triggers no solve.
                2020: replace(source, provider_anchor_year=2020),
                source.provider_anchor_year: source,
            },
        )
        service = "hydraulic_water_collection_row"
        result = _member_response(
            names,
            "remind_ssp2_npi",
            methods,
            registry,
            pd.DataFrame(
                [
                    {
                        "provider_service_id": service,
                        "provider_service_code": provider_service_code(service),
                    }
                ]
            ),
            providers,
        )
        direct = _direct_factor_table(names, methods, registry, result.direct_flows)
        # Native demand is per reference product, even for waste endpoints
        # whose source production is negative. No numeric database IDs persist.
        native = result.native_unit_responses
        assert set(zip(native.endpoint_database, native.endpoint_code, strict=True)) == {
            endpoint.key for endpoint in endpoints
        }
        assert native.source_anchor_year.eq(source.provider_anchor_year).all()
        assert native.source_database.eq("original" if original else "future").all()
        assert native.matrix_database.eq("frontier").all()
        assert not any(
            str(column).endswith("_id") and str(column).startswith(("endpoint", "product"))
            for column in native
        )
        assert set(native.endpoint_unit) == {endpoint.unit for endpoint in endpoints}
        assert len(result.provider_unit_responses) == len(methods)
        assert result.provider_unit_responses.provider_service_id.eq(service).all()
        for endpoint in endpoints:
            node = boundary.bd.get_node(database=endpoint.key[0], code=endpoint.key[1])
            product = node.rp_exchange().input
            rows = native.loc[
                native.endpoint_database.eq(endpoint.key[0])
                & native.endpoint_code.eq(endpoint.key[1])
            ]
            assert rows.product_database.eq(product["database"]).all()
            assert rows.product_code.eq(product["code"]).all()
        pam = boundary.bd.get_node(database=source.pam.key[0], code=source.pam.key[1])
        lca = bc.LCA({pam.rp_exchange().input.id: 1.0}, method_tuples[0])
        lca.lci()
        lca.lcia()
        row = native.loc[
            native.endpoint_database.eq(source.pam.key[0])
            & native.endpoint_code.eq(source.pam.key[1])
            & native.method_id.eq("z")
        ].iloc[0]
        assert row.unit_score == pytest.approx(lca.score)
        assert row.unit_contribution_norm == pytest.approx(abs(lca.characterized_inventory).sum())
        for method_id, factor in zip(("z", "a"), factors, strict=True):
            selected = direct.loc[direct.method_id.eq(method_id)].set_index("flow_code")
            assert selected.loc[flows[0].key[1], "factor"] == float(np.float32(factor))
            assert selected.loc[flows[1].key[1], "factor"] == 0.0
            assert bool(selected.loc[flows[1].key[1], "present"])
            assert selected.loc[flows[-1].key[1], "factor"] == 0.0
            assert not bool(selected.loc[flows[-1].key[1], "present"])
        assert len(direct) == len(flows) * len(methods)
        assert set(direct.endpoint_unit) == {flow.unit for flow in flows}
    # Prepared tables remain usable after the Brightway source project closes.
    for name, frame in (("native", native), ("direct", direct)):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
