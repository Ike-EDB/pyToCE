"""Complete chemical markets preserve local feed custody and native allocations."""

from math import fsum, log
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)

# Synthetic databases retain native market positions and source-derived or declared test quantities.
_MARKETS = {
    "methanol_delivered_cn": {
        "product": "methanol",
        "location": "CN",
        "manufacturer": "methanol",
        "suppliers": (
            ("methanol production, from coal gasification", "CN", 0.4982773471145564),
            ("methanol production, from natural gas reforming", "CN", 0.5017226528854436),
        ),
        "freight": (("barge", 0.0246), ("lorry", 0.1044), ("tanker", 0.599), ("rail", 0.1546)),
    },
    "methanol_delivered_rer": {
        "product": "methanol",
        "location": "RER w/o RU",
        "manufacturer": "methanol",
        "suppliers": tuple(
            ("methanol production, from natural gas reforming", location, amount)
            for location, amount in (
                ("LT", 0.07335452725774166),
                ("RS", 0.1130342635111268),
                ("DE", 0.2708112563287413),
                ("NL", 0.0365006475921347),
                ("NO", 0.5062993053102555),
            )
        ),
        "freight": (("barge", 0.021869035), ("lorry", 0.171298966), ("rail", 0.04381967)),
    },
    "p_xylene_delivered": {
        "product": "p-xylene",
        "location": "RER",
        "manufacturer": "p_xylene",
        "suppliers": (
            ("BTX production, from reformate, average", "RER", 0.8902870669862784),
            ("BTX production, from pyrolysis gas, average", "RER", 0.1097129330137216),
        ),
        "freight": (("tanker_barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "o_xylene_delivered": {
        "product": "o-xylene",
        "location": "RER",
        "manufacturer": "o_xylene",
        "suppliers": (
            ("BTX production, from reformate, average", "RER", 0.890279633457397),
            ("BTX production, from pyrolysis gas, average", "RER", 0.109720366542603),
        ),
        "freight": (("tanker_barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "glycerine_delivered": {
        "product": "glycerine",
        "location": "RER",
        "manufacturer": "glycerine",
        "suppliers": (
            (
                "treatment of used vegetable cooking oil, purified, esterification",
                "FR",
                0.02082937387058665,
            ),
            ("esterification of rape oil", "CH", 0.001256142358757409),
            ("glycerine production, from epichlorohydrin", "RER", 0.3324505228243788),
            ("esterification of rape oil", "Europe without Switzerland", 0.6454639609462771),
        ),
        "freight": (("barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "glucose_delivered": {
        "product": "glucose",
        "location": "GLO",
        "manufacturer": "glucose",
        "suppliers": (
            ("glucose production", "RER", 0.1740000000000001),
            ("glucose production", "RoW", 0.8259999999999998),
        ),
        "freight": (
            ("van", 0.0121),
            ("lorry", 0.3899),
            ("container_ship", 0.172),
            ("rail", 0.0994),
        ),
    },
    "cyclohexane_delivered": {
        "product": "cyclohexane",
        "location": "GLO",
        "manufacturer": "cyclohexane",
        # Synthetic quantities exercise two producer locations and four freight functions.
        "suppliers": (
            ("cyclohexane production, benzene hydrogenation, liquid phase process", "RoW", 0.65),
            ("cyclohexane production, benzene hydrogenation, liquid phase process", "RER", 0.35),
        ),
        "freight": (("barge", 0.03), ("lorry", 0.2), ("container_ship", 0.6), ("rail", 0.3)),
    },
    "hydrogen_cyanide_delivered": {
        "product": "hydrogen cyanide",
        "location": "RER",
        "manufacturer": "hydrogen_cyanide",
        # Every product-supplier row must change, including the three native
        # technologies other than the selected allocated Sohio coproduct.
        "suppliers": (
            ("hydrogen cyanide production, Shawinigan process", "RER", 0.2),
            ("hydrogen cyanide production, BMA process", "RER", 0.31),
            ("hydrogen cyanide production, Andrussow process", "RER", 0.34),
            ("acrylonitrile production, Sohio process", "RER", 0.15),
        ),
        "freight": (("lorry", 0.05), ("rail", 0.02)),
    },
    "acetylene_delivered": {
        "product": "acetylene",
        "location": "RER",
        "manufacturer": "acetylene",
        "suppliers": (
            ("acetylene production, from natural gas, operated with a water quench", "RER", 0.6),
            ("acetylene production, from natural gas, operated with an oil quench", "RER", 0.4),
        ),
        "freight": (("lorry", 0.06), ("rail", 0.01)),
    },
    "formaldehyde_delivered": {
        "product": "formaldehyde",
        "location": "RER",
        "manufacturer": "formaldehyde",
        # A small coproduct share must survive alongside the main producer.
        "suppliers": (
            ("dimethyl carbonate production", "RER", 0.009),
            ("formaldehyde production, methanol oxidation", "RER", 0.991),
        ),
        "freight": (("barge", 0.02), ("lorry", 0.17), ("rail", 0.04)),
    },
    "allyl_chloride_delivered": {
        "product": "allyl chloride",
        "location": "RER",
        "manufacturer": "allyl_chloride",
        "suppliers": (
            ("allyl chloride production, reaction of propylene and chlorine", "RER", 1.0),
        ),
        "freight": (("barge", 0.02), ("lorry", 0.17), ("rail", 0.04)),
    },
    "epichlorohydrin_delivered": {
        "product": "epichlorohydrin",
        "location": "GLO",
        "manufacturer": "epichlorohydrin",
        "suppliers": (
            ("epichlorohydrin production from allyl chloride", "RER", 0.3),
            ("epichlorohydrin production from allyl chloride", "RoW", 0.7),
        ),
        "freight": (("barge", 0.03), ("lorry", 0.2), ("container_ship", 0.6), ("rail", 0.3)),
    },
    "maleic_anhydride_delivered": {
        "product": "maleic anhydride",
        "location": "GLO",
        "manufacturer": "maleic_anhydride",
        "suppliers": (
            ("maleic anhydride production by catalytic oxidation of benzene", "RoW", 0.21),
            ("maleic anhydride production by direct oxidation of n-butane", "RER", 0.31),
            ("maleic anhydride production by direct oxidation of n-butane", "RoW", 0.4),
            ("maleic anhydride production by catalytic oxidation of benzene", "RER", 0.08),
        ),
        "freight": (("barge", 0.03), ("lorry", 0.2), ("container_ship", 0.6), ("rail", 0.3)),
    },
    "phosgene_delivered": {
        "product": "phosgene, liquid",
        "location": "RoW",
        "manufacturer": "phosgene",
        "suppliers": (("phosgene production, liquid", "RoW", 1.0),),
        # Invented amounts retain the native seven-row land-freight shape.
        # Future rows share two supplier keys but remain separate exchanges.
        "freight": (
            ("lorry_za", 0.001),
            ("lorry_br", 0.004),
            ("lorry_row", 0.05),
            ("rail_za", 0.002),
            ("rail_row", 0.006),
            ("rail_cn", 0.03),
            ("rail_in", 0.001),
        ),
    },
    "phenol_delivered": {
        "product": "phenol",
        "location": "RER",
        "manufacturer": "phenol",
        "suppliers": (("phenol production, cumene oxidation", "RER", 1.0),),
        "freight": (("barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "acetone_delivered": {
        "product": "acetone, liquid",
        "location": "RER",
        "manufacturer": "acetone",
        # Retain the small acetic-acid coproduct, allocated phenol coproduct, and their original
        # coefficients.
        "suppliers": (
            ("acetic acid production, from n-butane oxidation", "RER", 0.002415796592889089),
            ("phenol production, cumene oxidation", "RER", 0.9476449109989913),
            ("acetone production, from isopropanol", "RER", 0.04993929240811969),
        ),
        "freight": (("barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "bisphenol_a_delivered_rer": {
        "product": "bisphenol A, powder",
        "location": "RER",
        "manufacturer": "bisphenol_a",
        "suppliers": (("bisphenol A production, powder", "RER", 1.0),),
        "freight": (("barge", 0.021869035), ("lorry", 0.171298966), ("rail", 0.04381967)),
    },
    "bisphenol_a_delivered_row": {
        "product": "bisphenol A, powder",
        "location": "RoW",
        "manufacturer": "bisphenol_a",
        "suppliers": (
            ("bisphenol A production, powder", "KR", 0.1721524007420716),
            ("bisphenol A production, powder", "RoW", 0.7451244325745486),
            ("bisphenol A production, powder", "JP", 0.08272316668337976),
        ),
        "freight": (
            ("barge", 0.0246),
            ("lorry_za", 0.0037411136550834123),
            ("lorry_br", 0.02389524385491526),
            ("lorry_row", 0.18116364249000133),
            ("container_ship", 0.599),
            ("rail_za", 0.007952563395159033),
            ("rail_row", 0.15638662735739703),
            ("rail_cn", 0.11335517220735361),
            ("rail_in", 0.031405637040090285),
        ),
    },
    "aniline_delivered": {
        "product": "aniline",
        "location": "RER",
        "manufacturer": "aniline",
        "suppliers": (("aniline production", "RER", 1.0),),
        "freight": (("barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "toluene_delivered": {
        "product": "toluene, liquid",
        "location": "RER",
        "manufacturer": "toluene",
        "suppliers": (
            ("BTX production, from reformate, average", "RER", 0.7522568305341825),
            ("BTX production, from pyrolysis gas, average", "RER", 0.2477431694658174),
        ),
        "freight": (("barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "cumene_delivered": {
        "product": "cumene",
        "location": "RER",
        "manufacturer": "cumene",
        "suppliers": (
            (
                "p-diisopropylbenzene purification, distillation of crude p-diisopropylbenzene",
                "RER",
                0.0001954816388824455,
            ),
            ("cumene production, benzene alkylation, UOP process", "RER", 0.2499511295902794),
            ("cumene production, benzene alkylation, zeolite process", "RER", 0.7498533887708382),
        ),
        "freight": (("barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "phthalic_anhydride_delivered": {
        "product": "phthalic anhydride",
        "location": "RER",
        "manufacturer": "phthalic_anhydride",
        "suppliers": (("phthalic anhydride production, o-xylene oxidation", "RER", 1.0),),
        "freight": (("barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
    "nitrobenzene_delivered": {
        "product": "nitrobenzene",
        "location": "RER",
        "manufacturer": "nitrobenzene",
        "suppliers": (("nitrobenzene production", "RER", 1.0),),
        "freight": (("barge", 0.0219), ("lorry", 0.1713), ("rail", 0.0438)),
    },
}

_RECIPIENTS = {
    "ethylene": {"methanol_delivered_cn": 0.9415419111428915},
    "propylene": {"methanol_delivered_cn": 0.8958135208792372},
    "formaldehyde": {"methanol_delivered_rer": 1.0486288848263254},
    "methyl_methacrylate": {
        "methanol_delivered_rer": 0.29507348416882995,
        "hydrogen_cyanide_delivered": 0.28,
        "acetone_delivered": 0.6026643674761566,
    },
    "purified_terephthalic_acid": {"p_xylene_delivered": 0.658542674032142},
    "phthalic_anhydride": {"o_xylene_delivered": 0.8095879423223575},
    "polyether_polyols_long_chain": {"glycerine_delivered": 0.023053762},
    "polyether_polyols_short_chain": {
        "glycerine_delivered": 0.035900366,
        "glucose_delivered": 0.142213149,
    },
    # Unequal invented doses expose lost or duplicate procurement; source
    # coefficients remain read dynamically by the production owner.
    "cyclohexanone": {"cyclohexane_delivered": 0.93},
    "adipic_acid": {"cyclohexane_delivered": 0.67},
    "adiponitrile": {"hydrogen_cyanide_delivered": 0.53},
    "butanediol": {"acetylene_delivered": 0.32, "formaldehyde_delivered": 0.73},
    "epichlorohydrin": {"allyl_chloride_delivered": 0.74},
    "glycerine": {"epichlorohydrin_delivered": 1.02},
    "epoxy_resin": {
        "epichlorohydrin_delivered": 0.54,
        "bisphenol_a_delivered_rer": 0.670616894771728,
    },
    "succinic_acid": {"maleic_anhydride_delivered": 0.87},
    "unsaturated_polyester_resin": {
        "maleic_anhydride_delivered": 0.16,
        "phthalic_anhydride_delivered": 0.24,
    },
    "methylene_diphenyl_diisocyanate": {
        "formaldehyde_delivered": 0.14,
        "aniline_delivered": 0.7157881402721398,
    },
    "polycarbonate": {"phosgene_delivered": 0.4, "bisphenol_a_delivered_row": 0.925185597},
    "dinitrotoluene": {"toluene_delivered": 0.535177675},
    "toluene_diisocyanate": {"toluene_delivered": 0.23394131057642298},
    "phenol": {"cumene_delivered": 0.90072845220876},
    "acetone": {"cumene_delivered": 0.7297457608279803},
    "bisphenol_a": {"acetone_delivered": 0.262213668, "phenol_delivered": 0.845534432},
    "aniline": {"nitrobenzene_delivered": 1.2961226670708403},
}
_RECIPIENT_FAMILIES = {
    "methanol": ("ethylene", "propylene", "formaldehyde"),
    "other_feeds": (
        "purified_terephthalic_acid",
        "phthalic_anhydride",
        "polyether_polyols_long_chain",
        "polyether_polyols_short_chain",
    ),
    "pa_precursors": ("cyclohexanone", "adipic_acid", "adiponitrile"),
    "c4_precursors": ("butanediol", "succinic_acid", "methylene_diphenyl_diisocyanate"),
    "chlorinated_precursors": ("epichlorohydrin", "polycarbonate"),
    "resins": ("epoxy_resin", "unsaturated_polyester_resin"),
    "synthetic_glycerine": ("glycerine",),
    # Each recipient is compiled through its package rule; manufacturer endpoints use separate
    # fixture identities.
    "aromatic_producers": ("phenol", "acetone", "aniline", "toluene_diisocyanate"),
    "aromatic_consumers": ("bisphenol_a", "methyl_methacrylate"),
    "dnt": ("dinitrotoluene",),
}
_DIRECT_COAL = {"ethylene": 1.412312866714337, "propylene": 1.3437202813188558}
_ORIGINS = ("biogenic", "atmospheric_dac")
# A small invented source excess represents the observed nonunit future
# supplier sums. Exact copied-row comparisons must reject renormalization.
_FUTURE_CYCLOHEXANE_SHARE_EXCESS = 2e-8
_QUANTITIES = (
    "amount",
    "uncertainty type",
    "loc",
    "scale",
    "minimum",
    "maximum",
    "negative",
    "allocation",
    "source_record",
)


def _freight_key(market: str, mode: str, ordinary: str) -> str:
    """Retain separate native freight rows that share future supplier pointers."""
    if market in {"phosgene_delivered", "bisphenol_a_delivered_row"} and ordinary == "dated":
        mode = mode.split("_", maxsplit=1)[0]
    return f"{market}_{mode}"


def _source_amount(amount: float, database: str) -> float:
    """Retain native and premise float32 source precision."""
    return float(np.float32(amount)) if database == "dated" else amount


@pytest.mark.parametrize(("year", "ordinary"), ((2025, "native"), (2050, "dated")))
@pytest.mark.parametrize("family", tuple(_RECIPIENT_FAMILIES))
def test_complete_feed_markets_preserve_each_local_recipient_boundary(
    tmp_path: Path, year: int, ordinary: str, family: str
) -> None:
    """Compile recipient and delivery recipes with synthetic suppliers.

    Nonunit outputs, uncertainty and distinct emissions expose altered
    allocation, missing suppliers and duplicate freight. Native supplier
    quantities, signs and exchange records remain explicit.
    """
    catalogue = load_provider_catalogue()
    services = {
        row["provider_service_id"]: row for row in cast(list[dict[str, Any]], catalogue["services"])
    }
    # Biogenic glycerine uses esterification; only its synthetic DAC route
    # consumes epichlorohydrin. All other selected recipients cover both origins.
    origins = ("atmospheric_dac",) if family == "synthetic_glycerine" else _ORIGINS
    recipients = {stem: _RECIPIENTS[stem] for stem in _RECIPIENT_FAMILIES[family]}
    markets = {
        stem: _MARKETS[stem]
        for stem in dict.fromkeys(m for feeds in recipients.values() for m in feeds)
    }
    root_ids = tuple(f"{stem}_{origin}" for stem in recipients for origin in origins)
    delivery_ids = tuple(f"{stem}_{origin}" for stem in markets for origin in origins)
    supplied_ids = set()
    for service_id in (*root_ids, *delivery_ids):
        supplied_ids.update(
            row["service_id"]
            for row in services[service_id]["construction"].get("replacements", [])
            if row["service_id"] not in delivery_ids
        )
    supplied_ids = tuple(sorted(supplied_ids))
    assert set(root_ids).isdisjoint(supplied_ids)
    selected = [services[service_id] for service_id in (*root_ids, *delivery_ids)]
    for service_id in supplied_ids:
        selected.append(
            {
                **services[service_id],
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": {
                        "name": f"admitted {service_id}",
                        "reference_product": services[service_id]["reference_product"],
                        "unit": services[service_id]["unit"],
                        "exact_location": services[service_id]["location"],
                    },
                    "amount": 1.0,
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity=f"invented_remaining_{family}_delivery",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )

    # Unrelated feed and gas substitutions retain their actual declarations;
    # small external supplies cover their remaining input requirements.
    other_feeds: dict[str, list[tuple[dict[str, Any], float]]] = {}
    gas_flows: dict[tuple[str, tuple[str, ...]], str] = {}
    for stem, feeds in recipients.items():
        construction = services[f"{stem}_{origins[0]}"]["construction"]
        products = {_MARKETS[market]["product"] for market in feeds}
        other_feeds[stem] = [
            (
                rule,
                0.6148394521849897
                if stem == "toluene_diisocyanate" and rule["replacement_id"] == "dinitrotoluene"
                else 0.07 * (index + 1),
            )
            for index, rule in enumerate(construction["replace_technosphere"])
            if rule["selector"]["reference_product"] not in products
        ]
        for rule in construction.get("replace_biosphere", []):
            for label in ("selector", "replacement_selector"):
                selector = rule[label]
                key = (selector["name"], tuple(selector["categories"]))
                if key not in gas_flows:
                    gas_flows[key] = f"gas_{len(gas_flows)}"
    freight_codes = tuple(
        dict.fromkeys(
            _freight_key(market, mode, database)
            for market, data in markets.items()
            for mode, _amount in data["freight"]
            for database in ("native", "dated")
        )
    )
    markers = (*supplied_ids, *freight_codes, "unreplaced_feed", "utility", "direct")

    def inventory(database: str) -> dict[tuple[str, str], dict[str, Any]]:
        """Build complete market rows and nonunit recipients in a disposable store."""
        future = database == "dated"
        root_basis, market_basis, manufacturer_basis = (
            (3.0, 7.0, 5.0) if future else (2.0, 4.0, 2.0)
        )

        def exchange(
            code: str, amount: float, record: str, kind: str = "technosphere"
        ) -> dict[str, Any]:
            """Supply independent native uncertainty and identifiable allocation rows."""
            row = {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "amount": amount,
                "type": kind,
                "uncertainty type": 2,
                "loc": log(abs(amount)) + 0.03,
                "scale": 0.125,
                "minimum": abs(amount) * 0.8,
                "maximum": abs(amount) * 1.2,
                "negative": amount < 0,
                "allocation": "invented native allocation",
                "source_record": record,
                "activity": f"native producer identity {code}",
                "properties": {
                    "dry mass": {"amount": 1.0, "unit": "kg"},
                    **(
                        {
                            "carbon content, fossil": {"amount": 0.0, "unit": "dimensionless"},
                            "carbon content, non-fossil": {"amount": 0.4, "unit": "dimensionless"},
                        }
                        if code.startswith("glucose_delivered")
                        else {}
                    ),
                },
            }
            if future:
                row.update({"uncertainty type": 0, "loc": amount})
                for field in ("scale", "minimum", "maximum", "negative", "properties"):
                    del row[field]
            return row

        records = {}
        for index, service_id in enumerate(supplied_ids, start=1):
            records[database, service_id] = _process(
                database,
                service_id,
                f"admitted {service_id}",
                services[service_id]["reference_product"],
                location=services[service_id]["location"],
                production_amount=manufacturer_basis,
                exchanges=[
                    exchange(service_id, index * (3 if future else 1), service_id, "biosphere")
                ],
            )
        for market, data in markets.items():
            inputs = []
            for index, (name, location, share) in enumerate(data["suppliers"]):
                share = _source_amount(share, database)
                if future and market == "cyclohexane_delivered" and index == 0:
                    share += _FUTURE_CYCLOHEXANE_SHARE_EXCESS
                code = f"{market}_supplier_{index}"
                records[database, code] = _process(
                    database,
                    code,
                    name,
                    data["product"],
                    location=location,
                    exchanges=[exchange("unreplaced_feed", 16.0, code, "biosphere")],
                )
                inputs.append(exchange(code, share * market_basis, code))
            for index, (mode, dose) in enumerate(data["freight"], start=1):
                code = _freight_key(market, mode, database)
                # Shared future pointers have one inventory per supplier, while
                # each original freight row keeps its own amount and identity.
                marker_amount = (
                    0.25
                    if market in {"phosgene_delivered", "bisphenol_a_delivered_row"}
                    else index / 4
                )
                records[database, code] = _process(
                    database,
                    code,
                    f"transport, freight, {mode}",
                    f"transport, freight, {mode}",
                    location="GLO",
                    exchanges=[exchange(code, marker_amount, code, "biosphere")],
                )
                records[database, code]["unit"] = "ton kilometer"
                inputs.append(
                    exchange(
                        code, _source_amount(dose, database) * market_basis, f"{market}_{mode}"
                    )
                )
            records[database, market] = _process(
                database,
                market,
                f"market for {data['product']}",
                data["product"],
                location=data["location"],
                production_amount=market_basis,
                exchanges=inputs,
            )
        records[database, "utility"] = _process(
            database,
            "utility",
            "native recipient utility",
            "native recipient utility",
            exchanges=[exchange("utility", 0.7, "utility", "biosphere")],
        )
        for stem, feeds in recipients.items():
            construction = services[f"{stem}_{origins[0]}"]["construction"]
            base = construction["base"]["selector"]
            inputs = [
                exchange(market, _source_amount(dose, database) * root_basis, market)
                for market, dose in feeds.items()
            ]
            if stem in _DIRECT_COAL:
                inputs.append(
                    exchange(
                        "methanol_delivered_cn_supplier_0",
                        _DIRECT_COAL[stem] * root_basis,
                        "direct_coal",
                    )
                )
            for index, (rule, dose) in enumerate(other_feeds[stem]):
                selector = rule["selector"]
                code = f"{stem}_other_{index}"
                records[database, code] = _process(
                    database,
                    code,
                    selector["name"],
                    selector["reference_product"],
                    location=(
                        selector["location"]
                        if "location" in selector
                        else selector["preferred_locations"][0]
                    ),
                    exchanges=[exchange("unreplaced_feed", 16.0, code, "biosphere")],
                )
                inputs.append(exchange(code, _source_amount(dose, database) * root_basis, code))
            for index, rule in enumerate(construction.get("replace_biosphere", []), start=1):
                selector = rule["selector"]
                code = gas_flows[selector["name"], tuple(selector["categories"])]
                gas = exchange(code, 0.013 * index * root_basis, code, "biosphere")
                gas["properties"] = {
                    "carbon content, fossil": {
                        "amount": 12 / (44 if "dioxide" in selector["name"] else 28),
                        "unit": "dimensionless",
                    },
                    "carbon content, non-fossil": {"amount": 0.0, "unit": "dimensionless"},
                }
                inputs.append(gas)
            inputs.extend(
                (
                    exchange("utility", 0.11 * root_basis, "utility"),
                    exchange("direct", 0.017 * root_basis, "direct", "biosphere"),
                )
            )
            records[database, stem] = _process(
                database,
                stem,
                base["name"],
                base["reference_product"],
                location=base["exact_location"],
                production_amount=root_basis,
                exchanges=inputs,
            )
        for code in (*recipients, *markets):
            production = records[database, code]["exchanges"][0]
            production.update(
                exchange(code, production["amount"], f"production_{code}", "production")
            )
        for record in records.values():
            for row in record["exchanges"]:
                if row["type"] in {"technosphere", "production"}:
                    supplier = records[row["input"]]
                    row["name"] = supplier["name"] if future else supplier["reference product"]
                    row["unit"] = supplier["unit"]
                    if future:
                        # Premise embeds supplier geography and product, notably
                        # KR/RoW/JP BPA rows that will buy the selected RER factory.
                        row.update(
                            product=supplier["reference product"], location=supplier["location"]
                        )
        return records

    names = BrightwayNames(tmp_path / "brightway", "remaining_feed_delivery", "biosphere", "native")
    with selected_brightway_project(names) as bd:
        flows = {
            ("biosphere", code): {
                "name": f"invented {code} inventory witness",
                "unit": "kilogram",
                "categories": ("air",),
                "type": "emission",
            }
            for code in markers
        }
        flows.update(
            {
                ("biosphere", code): {
                    "name": name,
                    "unit": "kilogram",
                    "categories": categories,
                    "type": "emission",
                }
                for (name, categories), code in gas_flows.items()
            }
        )
        bd.Database("biosphere").write(flows, searchable=False)
        for database in ("native", "dated"):
            bd.Database(database).write(inventory(database), searchable=False)
        before = {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=ordinary,
            technology_database="dated",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
        market_productions = {}
        for market, data in markets.items():
            native = bd.get_node(database=ordinary, code=market)
            native_production = native.rp_exchange()
            native_rows = {row["source_record"]: row for row in native.technosphere()}
            for origin in origins:
                delivered = bd.get_node(
                    database=frontier.database_name,
                    code=provider_service_code(f"{market}_{origin}"),
                )
                delivered_production = delivered.rp_exchange()
                market_productions[delivered.key] = delivered_production
                if market == "glucose_delivered" and origin == "biogenic":
                    # Biogenic purchases keep the original GLO manufacturer mix
                    # and freight behind one native-market demand.
                    assert delivered_production["amount"] == 1.0
                    assert [(row.input.key, row["amount"]) for row in delivered.technosphere()] == [
                        (native.key, 1.0)
                    ]
                    continue
                assert {key: delivered_production.get(key) for key in _QUANTITIES} == {
                    key: native_production.get(key) for key in _QUANTITIES
                }
                rows = list(delivered.technosphere())
                assert len(rows) == len(data["suppliers"]) + len(data["freight"])
                assert {row["source_record"] for row in rows} == set(native_rows)
                for row in rows:
                    source = native_rows[row["source_record"]]
                    assert {key: row[key] for key in _QUANTITIES if key in row} == {
                        key: source[key] for key in _QUANTITIES if key in source
                    }
                    if "_supplier_" in row["source_record"]:
                        assert row.input.key == (
                            frontier.database_name,
                            provider_service_code(f"{data['manufacturer']}_{origin}"),
                        )
                        assert "activity" not in row
                        target = services[f"{data['manufacturer']}_{origin}"]
                        assert row["name"] == target["reference_product"]
                        assert "reference product" not in row
                        assert row["unit"] == target["unit"]
                        if ordinary == "dated":
                            assert row["product"] == target["reference_product"]
                            assert row["location"] == target["location"]
                            assert "properties" not in row
                        else:
                            assert "product" not in row and "location" not in row
                            assert row["properties"]["dry mass"] == source["properties"]["dry mass"]
                    else:
                        assert {
                            key: value for key, value in dict(row).items() if key != "output"
                        } == {key: value for key, value in dict(source).items() if key != "output"}

        for stem, feeds in recipients.items():
            native = bd.get_node(database=ordinary, code=stem)
            native_production = native.rp_exchange()
            native_rows = {row["source_record"]: row for row in native.technosphere()}
            native_biosphere = {row["source_record"]: row for row in native.biosphere()}
            for origin in origins:
                root_id = f"{stem}_{origin}"
                root = bd.get_node(
                    database=frontier.database_name, code=provider_service_code(root_id)
                )
                root_production = root.rp_exchange()
                rows = {row["source_record"]: row for row in root.technosphere()}
                assert set(rows) == set(native_rows)
                assert {key: root_production.get(key) for key in _QUANTITIES} == {
                    key: native_production.get(key) for key in _QUANTITIES
                }
                biosphere = {row["source_record"]: row for row in root.biosphere()}
                assert set(biosphere) == set(native_biosphere)
                for record, row in biosphere.items():
                    assert {key: row[key] for key in _QUANTITIES if key in row} == {
                        key: native_biosphere[record][key]
                        for key in _QUANTITIES
                        if key in native_biosphere[record]
                    }
                expected_manufacture = {service_id: 0.0 for service_id in supplied_ids}
                # The selected background supplies its matching freight functions.
                expected = {
                    code: 0.0 for code in (*supplied_ids, "unreplaced_feed", "utility", "direct")
                }
                expected.update(
                    {
                        _freight_key(market, mode, ordinary): 0.0
                        for market, data in markets.items()
                        for mode, _amount in data["freight"]
                    }
                )
                expected.update(utility=0.11 * 0.7, direct=0.017)
                for record, row in rows.items():
                    assert {key: row[key] for key in _QUANTITIES if key in row} == {
                        key: native_rows[record][key]
                        for key in _QUANTITIES
                        if key in native_rows[record]
                    }
                for market, dose in feeds.items():
                    dose = _source_amount(dose, ordinary)
                    assert rows[market].input.key == (
                        frontier.database_name,
                        provider_service_code(f"{market}_{origin}"),
                    )
                    assert rows[market]["amount"] / root_production["amount"] == pytest.approx(dose)
                    data = markets[market]
                    target = f"{data['manufacturer']}_{origin}"
                    supplied_share = fsum(
                        _source_amount(share, ordinary)
                        for _name, _location, share in data["suppliers"]
                    )
                    if ordinary == "dated" and market == "cyclohexane_delivered":
                        supplied_share += _FUTURE_CYCLOHEXANE_SHARE_EXCESS
                    if market == "glucose_delivered" and origin == "biogenic":
                        # Both unchanged native glucose suppliers carry the
                        # fixture's 16-unit marker per kg manufacture.
                        expected["unreplaced_feed"] += dose * supplied_share * 16.0
                    else:
                        expected_manufacture[target] += dose * supplied_share
                    for index, (mode, amount) in enumerate(data["freight"], start=1):
                        marker_amount = (
                            0.25
                            if market in {"phosgene_delivered", "bisphenol_a_delivered_row"}
                            else index / 4
                        )
                        expected[_freight_key(market, mode, ordinary)] += (
                            dose * _source_amount(amount, ordinary) * marker_amount
                        )
                if stem in _DIRECT_COAL:
                    assert rows["direct_coal"].input.key == (
                        frontier.database_name,
                        provider_service_code(f"methanol_{origin}"),
                    )
                    expected_manufacture[f"methanol_{origin}"] += _DIRECT_COAL[stem]
                replacements = {
                    item["replacement_id"]: item["service_id"]
                    for item in services[root_id]["construction"]["replacements"]
                }
                for index, (rule, dose) in enumerate(other_feeds[stem]):
                    target = replacements[rule["replacement_id"]]
                    assert rows[f"{stem}_other_{index}"].input.key == (
                        frontier.database_name,
                        provider_service_code(target),
                    )
                    expected_manufacture[target] += _source_amount(dose, ordinary)
                    if (
                        stem == "toluene_diisocyanate"
                        and rule["replacement_id"] == "dinitrotoluene"
                    ):
                        assert target == f"dinitrotoluene_{origin}"
                        assert rows[f"{stem}_other_{index}"]["amount"] == (
                            _source_amount(0.6148394521849897, ordinary) * root_production["amount"]
                        )
                if stem == "polyether_polyols_short_chain" and ordinary == "native":
                    glucose = rows["glucose_delivered"]["properties"]
                    if origin == "biogenic":
                        assert glucose == native_rows["glucose_delivered"]["properties"]
                    else:
                        assert "carbon content, fossil" not in glucose
                        assert "carbon content, non-fossil" not in glucose
                assert rows["utility"].input.key == (ordinary, "utility")
                assert {
                    key: value for key, value in dict(rows["utility"]).items() if key != "output"
                } == {
                    key: value
                    for key, value in dict(native_rows["utility"]).items()
                    if key != "output"
                }
                calculation = LCA({root: 1.0})
                calculation.lci()
                assert cast(NDArray[np.float64], calculation.supply_array)[
                    calculation.dicts.activity[root.id]
                ] == pytest.approx(1 / native_production["amount"])
                for market, dose in feeds.items():
                    dose = _source_amount(dose, ordinary)
                    delivered = rows[market].input
                    assert cast(NDArray[np.float64], calculation.supply_array)[
                        calculation.dicts.activity[delivered.id]
                    ] == pytest.approx(dose / market_productions[delivered.key]["amount"])
                    freight_demands: dict[str, float] = {}
                    for mode, amount in markets[market]["freight"]:
                        code = _freight_key(market, mode, ordinary)
                        freight_demands[code] = freight_demands.get(
                            code, 0.0
                        ) + dose * _source_amount(amount, ordinary)
                    for code, amount in freight_demands.items():
                        freight = bd.get_node(database=ordinary, code=code)
                        assert cast(NDArray[np.float64], calculation.supply_array)[
                            calculation.dicts.activity[freight.id]
                        ] == pytest.approx(amount)
                for index, service_id in enumerate(supplied_ids, start=1):
                    expected[service_id] = (
                        expected_manufacture[service_id]
                        * index
                        * (3 / 5 if ordinary == "dated" else 1 / 2)
                    )
                for code, amount in expected.items():
                    flow = bd.get_node(database="biosphere", code=code)
                    actual = float(
                        calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum()
                    )
                    assert actual == pytest.approx(amount, rel=3e-7)
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
