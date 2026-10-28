"""Activity years and dated suppliers in synthetic LCA timelines."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from infiltration_fixtures import resolved_infiltration_provider
from material_carbon_fixtures import prepared_material_carbon

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.run_lca.calculation.providers import (
    load_provider_routes,
)
from pytoce.run_lca.calculation.timex import (
    _provider_activations,
    database_dates,
)
from tests.integration.brightway_lca_fixtures import PROVIDER_TECHNICAL_YEARS, write_prepared_inputs


def test_database_dates_select_prepared_provider_service_frontiers() -> None:
    """Keep the foreground dynamic and assign prepared years to provider databases."""
    frontiers = {year: f"selected_provider_services_{year}" for year in PROVIDER_TECHNICAL_YEARS}
    observed = database_dates(
        "foreground",
        frontiers,
    )

    assert observed == {
        "foreground": "dynamic",
        **{database: datetime(year, 1, 1) for year, database in frontiers.items()},
    }
    reversed_dates = {
        date: database for database, date in observed.items() if isinstance(date, datetime)
    }
    assert reversed_dates == {
        datetime(year, 1, 1): frontiers[year] for year in PROVIDER_TECHNICAL_YEARS
    }


def test_provider_routes_are_loaded_from_the_shared_prepared_product(tmp_path: Path) -> None:
    """Read synthetic material carbon data and resolve the packaged provider routes."""
    carbon = prepared_material_carbon(tmp_path / "native_carbon")
    provider = resolved_infiltration_provider(tmp_path / "native_infiltration")
    provider_identity = provider.identity
    shared_manifest, _ = write_prepared_inputs(
        tmp_path,
        BrightwayNames(
            tmp_path / "data_processed/brightway", "route_fixture", "biosphere", "native"
        ),
        methods=pd.DataFrame(),
        provider=provider,
        carbon_rows=carbon,
        backgrounds=(),
    )
    loaded = load_provider_routes(shared_manifest)

    assert loaded.identity == provider_identity
    assert loaded.interface_database == provider.interface_database
    assert set(loaded.material_carbon.constituents) == {
        (row["material"], row["constituent"]) for row in carbon
    }
    assert loaded.material_carbon.fractions[
        loaded.material_carbon.constituents.index(("HDPE", "hdpe"))
    ] == ((6 / 7, 0.0, 0.0), (0.0, 6 / 7, 0.0), (0.0, 0.0, 6 / 7))
    request = {
        "provider_demand_type": "polymerization_operation",
        "material": "HDPE",
        "constituent": "hdpe",
        "product": "polymer",
        "virgin_feedstock_route": "fossil",
        "ultimate_carbon_origin": "fossil",
        "mapping_basis": "kg_destination_polymer_output",
        "year": 2050,
    }
    assert (
        loaded.resolve(**request)[0].provider_service_code
        == "provider_service__hdpe_polymerization_operation"
    )

    with pytest.raises(LookupError, match="No provider route"):
        loaded.resolve(**{**request, "product": "unmodeled_product"})


class _Node(dict):
    """Supply the Brightway node metadata and reference-product ID used in mapping."""

    def __init__(
        self,
        database: str,
        code: str,
        node_id: int,
        product_id: int,
        **metadata: str,
    ) -> None:
        """Store external node and product identifiers with their supplied metadata."""
        super().__init__(code=code, **metadata)
        self.key = (database, code)
        self.id = node_id
        self._product_id = product_id

    def rp_exchange(self) -> SimpleNamespace:
        """Return the reference-product exchange fields read by temporal mapping."""
        return SimpleNamespace(input=SimpleNamespace(id=self._product_id))


class _BrightwayFixture:
    """Provide synthetic Brightway nodes for the foreground and dated providers."""

    def __init__(self) -> None:
        """Define one European foreground and provider records for 2025 and 2030."""
        self.by_id = {
            1: _Node("provider_interface", "polymer_service", 1, 1),
            2: _Node("foreground", "root_eur", 2, 2, region="EUR"),
        }
        self.by_key = {
            ("provider_2025", "polymer_service"): _Node(
                "provider_2025",
                "polymer_service",
                10,
                100,
                provider_source_database="ecoinvent_3_12_cutoff",
                provider_source_role="native_ecoinvent_provider",
                general_background_database="ecoinvent_3_12_cutoff",
                general_background_anchor_year="2025",
                general_background_role="ordinary_ecoinvent",
                selected_background_member="remind_ssp2_npi",
            ),
            ("provider_2030", "polymer_service"): _Node(
                "provider_2030",
                "polymer_service",
                11,
                101,
                provider_source_database="remind_ssp2_npi_2030",
                provider_source_role="matching_member_premise_provider",
                general_background_database="remind_ssp2_npi_2030",
                general_background_anchor_year="2030",
                general_background_role="matching_member_premise_general_background",
                selected_background_member="remind_ssp2_npi",
            ),
        }

    def get_node(
        self,
        *,
        id: int | None = None,
        database: str | None = None,
        code: str | None = None,
    ) -> _Node:
        """Return a synthetic node by integer ID or by database and code."""
        if id is not None:
            return self.by_id[id]
        return self.by_key[(str(database), str(code))]


def test_timex_frontier_shares_preserve_detailed_activation_attribution() -> None:
    """Split a 2027 polymer demand between dated providers and retain its activity year.

    The synthetic timeline assigns 60% to 2025 and 40% to 2030. Detailed demands
    remain 3 kg fossil polyethylene and 7 kg biogenic polyethylene. The external
    node double supplies lookup and reference-product data to the package
    calculation. A demand in 2028 without a timeline mapping must fail.
    """
    calculation = SimpleNamespace(
        timeline=pd.DataFrame(
            [
                {
                    "producer": 1,
                    "consumer": 2,
                    "date_producer": pd.Timestamp("2027-01-01"),
                    "temporal_market_shares": {
                        "provider_2025": 0.6,
                        "provider_2030": 0.4,
                    },
                    "cumulative_amount": 10.0,
                }
            ]
        )
    )
    detailed = pd.DataFrame(
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
                "good": "all",
                "custody_status": "new_polymer_output",
                "ultimate_carbon_origin": "fossil",
                "branch_lineage_id": "fossil_branch",
                "basis_flow_names": "new_polymer => processing + new_polymer => primary_market",
                "lca_fate": "",
                "fate_origin": "",
                "amount": 3.0,
                "mfa_carbon_amount_kg": 3.0 * 6 / 7,
            },
            {
                "region": "EUR",
                "year": 2027,
                "provider_service_code": "polymer_service",
                "demand_type": "polymerization_operation",
                "stage_group": "material_production",
                "cutoff_cycle_role": "shared_polymerization",
                "physical_event_basis": "year new polymer is produced",
                "physical_event_year": 2027,
                "material": "BioPlastics",
                "good": "all",
                "custody_status": "new_polymer_output",
                "ultimate_carbon_origin": "biogenic",
                "branch_lineage_id": "biogenic_branch",
                "basis_flow_names": "new_polymer => processing + new_polymer => primary_market",
                "lca_fate": "",
                "fate_origin": "",
                "amount": 7.0,
                "mfa_carbon_amount_kg": 7.0 * 6 / 7,
            },
        ]
    )
    detailed["source_parent_amount_kg"] = detailed["amount"]
    detailed["constituent"] = ("hdpe", "bio_pe")
    detailed["product"] = "polymer"
    detailed["virgin_feedstock_route"] = "all"
    detailed["source_allocation_share"] = 1.0
    detailed["provider_demand_type"] = "polymerization_operation"
    detailed["mapping_basis"] = "kg_destination_polymer_output"
    detailed["lca_boundary"] = "native_polymerization_excluding_ethylene_and_propylene_inputs"
    detailed["burden_attribution_basis"] = "all_new_polymer_output"
    detailed["material_cycle_source"] = "new_polymer"
    detailed["origin_fraction"] = 1.0
    detailed["route_id"] = ("hdpe_operation", "bio_pe_operation")
    detailed["route_component_id"] = detailed["route_id"]
    detailed["provider_service_id"] = "hdpe_polymerization_operation"
    detailed["provider_role"] = "material_production"
    detailed["provider_material"] = "hdpe"
    detailed["infiltration_class"] = ""
    detailed["material_component_share"] = 1.0
    detailed["infiltration_share"] = 1.0
    detailed["provider_class"] = ("", "specified constituent polymer or grade")
    detailed["analogue_status"] = ("direct_match", "hdpe_for_unspecified_bio_pe")
    detailed["throughput_multiplier"] = 1.0
    detailed["resolution_status"] = ""
    detailed["material_functional_unit"] = ""
    detailed["styrene_component_share"] = 0.0
    detailed["butadiene_analogue_component_share"] = 0.0
    detailed["sbr_share_in_material"] = 0.0
    detailed["sbr_proxy_component_role"] = ""
    detailed["whole_tyre_constituents_excluded"] = False
    detailed["no_treatment_sensitivity_executed"] = False
    detailed["declared_provider_source_role"] = "native_ecoinvent_provider"
    activation_keys = {
        (str(region), int(year), str(service_code))
        for region, year, service_code in detailed[
            ["region", "year", "provider_service_code"]
        ].itertuples(index=False, name=None)
    }

    observed = _provider_activations(
        calculation.timeline,
        _BrightwayFixture(),
        detailed,
        {2025: "provider_2025", 2030: "provider_2030"},
        {2025: "provider_2025", 2030: "provider_2030"},
        activation_keys=activation_keys,
        producer_codes={},
        targets={},
    )

    amounts = observed.set_index(["provider_database", "constituent"])["amount"].to_dict()
    assert amounts == pytest.approx(
        {
            ("provider_2025", "hdpe"): 1.8,
            ("provider_2025", "bio_pe"): 4.2,
            ("provider_2030", "hdpe"): 1.2,
            ("provider_2030", "bio_pe"): 2.8,
        }
    )
    assert observed.groupby("provider_database")["amount"].sum().to_dict() == pytest.approx(
        {
            "provider_2025": 6.0,
            "provider_2030": 4.0,
        }
    )
    assert observed.groupby("provider_database")["mfa_carbon_amount_kg"].sum().to_dict() == (
        pytest.approx({"provider_2025": 6.0 * 6 / 7, "provider_2030": 4.0 * 6 / 7})
    )
    assert set(observed["stage_group"]) == {"material_production"}
    assert observed.groupby("constituent").amount.sum().to_dict() == pytest.approx(
        {"hdpe": 3.0, "bio_pe": 7.0}
    )
    assert observed["virgin_feedstock_route"].eq("all").all()
    assert observed["product"].eq("polymer").all()
    assert set(observed["cutoff_cycle_role"]) == {"shared_polymerization"}
    assert set(observed["physical_event_year"]) == {2027}
    assert set(observed["throughput_multiplier"]) == {1.0}
    assert set(observed["resolution_status"]) == {""}
    assert set(observed["provider_source_role"]) == {
        "native_ecoinvent_provider",
        "matching_member_premise_provider",
    }
    assert set(observed["general_background_role"]) == {
        "ordinary_ecoinvent",
        "matching_member_premise_general_background",
    }
    assert (
        observed.loc[observed["provider_anchor_year"].eq(2025), "general_background_anchor_year"]
        .eq(2025)
        .all()
    )

    unmapped = detailed.iloc[[0]].copy()
    unmapped["year"] = 2028
    unmapped["branch_lineage_id"] = "unmapped_branch"
    with_unmapped = pd.concat([detailed, unmapped], ignore_index=True)
    with_unmapped_keys = {
        (str(region), int(year), str(service_code))
        for region, year, service_code in with_unmapped[
            ["region", "year", "provider_service_code"]
        ].itertuples(index=False, name=None)
    }
    with pytest.raises(
        KeyError,
        match="Foreground activations have no Timex provider mapping:.*2028",
    ):
        _provider_activations(
            calculation.timeline,
            _BrightwayFixture(),
            with_unmapped,
            {2025: "provider_2025", 2030: "provider_2030"},
            {2025: "provider_2025", 2030: "provider_2030"},
            activation_keys=with_unmapped_keys,
            producer_codes={},
            targets={},
        )
