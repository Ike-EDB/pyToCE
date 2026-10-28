"""Focused provider catalog authority contracts."""

from typing import Any, cast

from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
    _chemical_recycling_authority,
)


def test_chemical_recycling_input_operations_own_embedded_reject_treatment() -> None:
    """Normalize the source treatment boundary and retain embedded reject owners."""
    catalogue = cast(dict[str, Any], provider_catalogue.load_provider_catalogue())
    assert catalogue["schema_version"] == "pytoce_provider_catalogue_schema_v1"
    services = {str(service["provider_service_id"]): service for service in catalogue["services"]}
    authority = cast(dict[str, Any], _chemical_recycling_authority())
    assert authority["schema_version"] == "pytoce_chemical_recycling_authority_schema_v1"
    source_authority = authority["source_authorities"][0]
    assert source_authority["source_id"] == "ozoemena_coles_2023_hydrothermal_recycling"
    assert source_authority["title"] == (
        "Hydrothermal Treatment of Waste Plastics: An Environmental Impact Study"
    )
    assert source_authority["doi"] == "10.1007/s10924-023-02792-3"
    preparation = authority["stages"]["preparation"]
    conversion = authority["stages"]["conversion"]

    assert preparation["selector"] == {
        "name": "treatment of waste mixed polyolefins, unprocessed, sorting",
        "reference_product": "waste mixed polyolefins, unprocessed",
        "unit": "kilogram",
        "exact_location": "GB",
    }
    assert preparation["functional_unit"] == (
        "1 kg tracked plastic entering chemical-recycling preparation"
    )
    assert preparation["processed_output_selector"] == {
        "reference_product": "waste mixed polyolefins, processed",
        "unit": "kilogram",
        "exchange_type": "technosphere",
        "sign": "negative",
    }
    assert preparation["plastic_reject_output_selector"] == {
        "reference_product": "waste plastic, mixture",
        "unit": "kilogram",
        "exchange_type": "technosphere",
        "sign": "negative",
    }
    normalization = preparation["tracked_plastic_denominator"]
    assert normalization["included_output_classes"] == [
        "processed_plastic",
        "combustible_plastic_reject",
    ]
    assert normalization["excluded_output_classes"] == [
        "water",
        "ferrous_metal",
        "non_ferrous_metal",
        "inert_material",
        "other_source_classified_nonplastic_output",
    ]
    assert normalization["normalization_status"] == "tracked_plastic_functional_unit"
    assert conversion["reference_products"] == [
        "naphtha",
        "distillate gas oil",
        "heavy gas oil",
        "heavy wax residue",
    ]
    assert conversion["denominator_selector"]["sign"] == "positive"

    for carbon_class in ("fossil", "non_fossil"):
        preparation_service = services[f"chemical_recycling_preparation_{carbon_class}"]
        conversion_service = services[f"chemical_recycling_conversion_{carbon_class}"]
        assert preparation_service["construction"]["kind"] == (
            "chemical_recycling_preparation_operation"
        )
        assert conversion_service["construction"]["kind"] == (
            "chemical_recycling_conversion_operation"
        )
        assert preparation_service["stage_owner"] == "chemical_recycling_input_operations"
        assert conversion_service["stage_owner"] == "chemical_recycling_input_operations"
        assert conversion_service["construction"]["direct_treatment_carbon_mappings"] == [
            {
                "selector": {
                    "name": "Carbon monoxide, fossil",
                    "unit": "kilogram",
                    "categories": ["air"],
                },
                "replacement_selector": {
                    "name": "Carbon monoxide, non-fossil",
                    "unit": "kilogram",
                    "categories": ["air"],
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
        ]

    internal_ids = {
        "plastic_waste_market_gb_non_fossil",
        "plastic_waste_open_burning_glo_non_fossil",
        "plastic_waste_landfill_row_non_fossil",
        "plastic_waste_incineration_gb_non_fossil",
    }
    for service_id in internal_ids:
        assert services[service_id]["frontier_service"] is True
        assert services[service_id]["foreground_service"] is False
    assert (
        services["plastic_waste_market_gb_non_fossil"]["construction"]["kind"]
        == "waste_market_origin_operation"
    )
    treatment_kinds = {
        services[service_id]["construction"]["kind"]
        for service_id in internal_ids
        if service_id != "plastic_waste_market_gb_non_fossil"
    }
    assert treatment_kinds == {"treatment_origin_operation"}
    assert all(
        services[service_id]["construction"]["preserve_native_production"] is True
        for service_id in internal_ids
        if service_id != "plastic_waste_market_gb_non_fossil"
    )


def test_project_recipe_treatment_components_keep_native_negative_demand_sign() -> None:
    """Prevent waste-treatment calls from becoming technosphere credits."""

    catalogue = cast(dict[str, Any], provider_catalogue.load_provider_catalogue())
    expected = {
        ("polyamide_6_resin_biogenic", "wastewater_treatment"): -0.002,
        ("polyamide_6_resin_atmospheric_dac", "wastewater_treatment"): -0.002,
        (
            "caprolactam_biogenic",
            "mixed_wastewater_treatment",
        ): -0.06167081048091456,
        (
            "caprolactam_atmospheric_dac",
            "mixed_wastewater_treatment",
        ): -0.06167081048091456,
        ("polyamide_6_resin_biogenic", "hazardous_waste_treatment"): -0.00035,
        (
            "polyamide_6_resin_atmospheric_dac",
            "hazardous_waste_treatment",
        ): -0.00035,
        ("polyamide_66_resin_biogenic", "wastewater_treatment"): -0.00225,
        ("polyamide_66_resin_atmospheric_dac", "wastewater_treatment"): -0.00225,
        ("polyamide_66_resin_biogenic", "hazardous_waste_treatment"): -0.00035,
        (
            "polyamide_66_resin_atmospheric_dac",
            "hazardous_waste_treatment",
        ): -0.00035,
        (
            "lactic_acid_atmospheric_dac",
            "waste_gypsum_treatment",
        ): -0.9778952,
        (
            "lactic_acid_atmospheric_dac",
            "average_wastewater_treatment",
        ): -0.00603606,
        (
            "lactic_acid_atmospheric_dac",
            "biowaste_incineration",
        ): -0.4150826,
        ("pla_atmospheric_dac", "average_wastewater_treatment"): -0.00047505,
        ("pla_atmospheric_dac", "hazardous_waste_treatment"): -0.0058415,
        ("pla_fossil", "hazardous_waste_treatment"): -0.0058415,
        ("pla_fossil", "average_wastewater_treatment"): -0.00047505,
        (
            "ethanol_to_butadiene_b3_operation",
            "waste_zeolite_treatment",
        ): -0.00442 * 0.887952520414271,
        (
            "ethanol_to_butadiene_b3_operation",
            "hazardous_waste_underground_treatment",
        ): -0.0015 * 0.887952520414271,
        (
            "ethanol_to_butadiene_b3_operation",
            "vegetable_oil_refinery_wastewater_treatment",
        ): -0.0010867 * 0.887952520414271,
        (
            "ethanol_to_butadiene_b3_operation",
            "average_wastewater_treatment",
        ): -0.0067144 * 0.887952520414271,
        ("p_xylene_biogenic", "average_wastewater_treatment"): -0.00159431481226472,
        ("p_xylene_biogenic", "spent_zeolite_disposal"): -0.00129777355495704,
        ("p_xylene_atmospheric_dac", "average_wastewater_treatment"): -0.00159431481226472,
        ("p_xylene_atmospheric_dac", "spent_zeolite_disposal"): -0.00129777355495704,
        ("o_xylene_biogenic", "average_wastewater_treatment"): -0.0015858008813036228,
        ("o_xylene_biogenic", "spent_zeolite_disposal"): -0.0012908432082243601,
        ("o_xylene_atmospheric_dac", "average_wastewater_treatment"): -0.0015858008813036228,
        ("o_xylene_atmospheric_dac", "spent_zeolite_disposal"): -0.0012908432082243601,
        (
            "phb_biogenic",
            "residual_bacterial_biomass_waste_to_energy_analogue",
        ): -0.42,
        (
            "phb_atmospheric_dac",
            "residual_bacterial_biomass_waste_to_energy_analogue",
        ): -0.42,
        (
            "phb_fossil",
            "residual_bacterial_biomass_waste_to_energy_analogue",
        ): -0.014227642276422764,
        ("pbat_biogenic", "reaction_water_treatment"): -0.00017212497471760035,
        (
            "pbat_atmospheric_dac",
            "reaction_water_treatment",
        ): -0.00017212497471760035,
        ("pbat_fossil", "reaction_water_treatment"): -0.00017212497471760035,
        ("pbat_biogenic", "tetrahydrofuran_purge_treatment"): -0.00944,
        ("pbat_atmospheric_dac", "tetrahydrofuran_purge_treatment"): -0.00944,
        ("pbat_fossil", "tetrahydrofuran_purge_treatment"): -0.00944,
        (
            "others_recycled_glass_bead_manufacturing",
            "glass_dust_landfill",
        ): -0.009995202302895,
    }
    # Quantitative recovery and extrusion equations are checked at their
    # biochemical owner; this test retains their negative-demand contract.
    biochemical_treatment = {
        ("starch_atmospheric_dac", "aqueous_residue_treatment"),
        ("starch_fossil", "aqueous_residue_treatment"),
    }
    treatment_components = expected.keys() | biochemical_treatment

    def components(value: object) -> list[dict[str, object]]:
        """Collect the recipe components, including nested process operations."""
        found: list[dict[str, object]] = []
        if isinstance(value, dict):
            found.extend(value.get("components", []))
            for nested in value.values():
                found.extend(components(nested))
        elif isinstance(value, list):
            for nested in value:
                found.extend(components(nested))
        return found

    actual = {
        (str(service["provider_service_id"]), str(component["component_id"])): float(
            cast(float, component["amount"])
        )
        for service in catalogue["services"]
        for component in components(service["construction"])
        if (service["provider_service_id"], component.get("component_id")) in treatment_components
    }
    assert all(actual[key] < 0.0 for key in biochemical_treatment)
    assert {
        key: amount for key, amount in actual.items() if key not in biochemical_treatment
    } == expected
