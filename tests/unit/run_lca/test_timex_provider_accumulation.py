"""Timex provider mapping and foreground coverage."""

from types import SimpleNamespace

import pandas as pd
import pytest

from pytoce.run_lca.calculation.timex import _provider_activations


class _Node(dict[str, object]):
    """Small Brightway node with the public fields read by the Timex boundary."""

    def __init__(
        self,
        database: str,
        code: str,
        node_id: int,
        product_id: int,
        **metadata: object,
    ) -> None:
        super().__init__(code=code, **metadata)
        self.key = (database, code)
        self.id = node_id
        self._product_id = product_id

    def rp_exchange(self) -> SimpleNamespace:
        """Return the reference-product exchange used to identify the product."""

        return SimpleNamespace(input=SimpleNamespace(id=self._product_id))


class _BrightwayBoundary:
    """Resolve temporal producers, roots, and dated provider products."""

    def __init__(self, *, anchor_year: int) -> None:
        metadata: dict[str, object] = {
            "provider_source_database": "source",
            "provider_source_role": "native_ecoinvent_provider",
            "general_background_database": (
                "ecoinvent_3_12_cutoff" if anchor_year <= 2025 else "remind_ssp2_npi_2030"
            ),
            "general_background_anchor_year": 2025 if anchor_year <= 2025 else anchor_year,
            "general_background_role": (
                "ordinary_ecoinvent"
                if anchor_year <= 2025
                else "matching_member_premise_general_background"
            ),
            "selected_background_member": "remind_ssp2_npi",
        }
        self.by_id = {
            1: _Node("provider_interface", "polymer_service", 1, 1),
            2: _Node("foreground", "root_eur", 2, 2, region="EUR"),
        }
        self.by_key = {
            (f"provider_{anchor_year}", "polymer_service"): _Node(
                f"provider_{anchor_year}",
                "polymer_service",
                10,
                100,
                **metadata,
            )
        }

    def get_node(
        self,
        *,
        id: int | None = None,
        database: str | None = None,
        code: str | None = None,
    ) -> _Node:
        """Return one node by the public Brightway identifier or database key."""

        if id is not None:
            return self.by_id[id]
        return self.by_key[(str(database), str(code))]


def _activation() -> pd.DataFrame:
    """Return one fully described foreground provider activation."""

    return pd.DataFrame(
        [
            {
                "region": "EUR",
                "year": 2027,
                "provider_service_code": "polymer_service",
                "demand_type": "polymerization_operation",
                "stage_group": "material_production",
                "cutoff_cycle_role": "shared_polymerization",
                "physical_event_basis": "year new polymer is produced",
                "physical_event_year": 2027,
                "material": "HDPE",
                "constituent": "hdpe",
                "product": "polymer",
                "virgin_feedstock_route": "all",
                "good": "all",
                "custody_status": "new_polymer_output",
                "ultimate_carbon_origin": "fossil",
                "branch_lineage_id": "fossil_branch",
                "basis_flow_names": "new_polymer => processing + new_polymer => primary_market",
                "lca_fate": "",
                "fate_origin": "",
                "amount": 3.0,
                "mfa_carbon_amount_kg": 3.0 * 6 / 7,
                "source_parent_amount_kg": 3.0,
                "source_allocation_share": 1.0,
                "provider_demand_type": "polymerization_operation",
                "mapping_basis": "kg_destination_polymer_output",
                "lca_boundary": "native_polymerization_excluding_ethylene_and_propylene_inputs",
                "burden_attribution_basis": "all_new_polymer_output",
                "material_cycle_source": "new_polymer",
                "origin_fraction": 1.0,
                "route_id": "polymer_route",
                "route_component_id": "polymer_component",
                "provider_service_id": "hdpe_polymerization_operation",
                "provider_role": "material_production",
                "provider_material": "hdpe",
                "infiltration_class": "",
                "material_component_share": 1.0,
                "infiltration_share": 1.0,
                "provider_class": "",
                "analogue_status": "direct_match",
                "throughput_multiplier": 1.0,
                "resolution_status": "",
                "material_functional_unit": "",
                "styrene_component_share": 0.0,
                "butadiene_analogue_component_share": 0.0,
                "sbr_share_in_material": 0.0,
                "sbr_proxy_component_role": "",
                "whole_tyre_constituents_excluded": False,
                "no_treatment_sensitivity_executed": False,
                "declared_provider_source_role": "native_ecoinvent_provider",
            }
        ]
    )


def _timeline(anchor_year: int) -> pd.DataFrame:
    """Return a Timex temporal-market row for one dated provider frontier."""

    return pd.DataFrame(
        [
            {
                "producer": 1,
                "consumer": 2,
                "date_producer": pd.Timestamp("2027-01-01"),
                "temporal_market_shares": {f"provider_{anchor_year}": 1.0},
                "cumulative_amount": 3.0,
            }
        ]
    )


def _resolve(
    bd: _BrightwayBoundary,
    *,
    anchor_year: int,
    activation_keys: set[tuple[str, int, str]] | None = None,
) -> pd.DataFrame:
    """Run the provider mapper with one prepared frontier and foreground row."""

    return _provider_activations(
        _timeline(anchor_year),
        bd,
        _activation(),
        {anchor_year: f"provider_{anchor_year}"},
        {anchor_year: f"provider_{anchor_year}"},
        activation_keys=(
            {("EUR", 2027, "polymer_service")} if activation_keys is None else activation_keys
        ),
        producer_codes={},
        targets={},
    )


def test_timex_provider_requires_a_matching_foreground_activation() -> None:
    """Reject a temporal-market producer that has no foreground amount."""

    with pytest.raises(KeyError, match="without a matching foreground activation"):
        _resolve(_BrightwayBoundary(anchor_year=2025), anchor_year=2025, activation_keys=set())
