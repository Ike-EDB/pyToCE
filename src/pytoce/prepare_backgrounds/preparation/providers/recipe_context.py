"""Prepared database resolvers and hybrid-provider recipe context."""

import json
import math
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, cast

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe

from .native_resolution import _ActivityResolver, _exchange_record, _FlowResolver, _input_semantics


@dataclass(slots=True)
class _RecipeContext:
    """Compile dated study inventories or read original native reference coefficients.

    Attributes:
        background_scenario: Selected REMIND member, or None for original reference admission.
        year: Provider-technology anchor year.
        provider_database: Project database receiving compiled provider activities.
        source: Resolver for original ecoinvent reference inventories and properties.
        ordinary: Resolver for the general background at the selected anchor.
        technology: Resolver for selected provider technologies or the original
            native reference.
        biosphere: Resolver for ecoinvent 3.12 elementary flows.
        catalogue: Validated provider recipe catalog.
        material_recipe: Admitted manufacturing, recovery and native dry-feed bases.
        method_authorities: Method source records indexed by identifier.
        complete_source_ids: Complete source references for each provider service.
        semantic_rows: Mutable records describing each scientific mapping.
        technology_rows: Mutable records describing early premise technology copies.
        traced_services: Provider services already recorded in ``semantic_rows``.
        native_exchanges: Cached native exchanges by Brightway activity key.
        input_nodes: Cached exchange inputs by Brightway node key.
        hybrid_datasets: Early premise activities copied into the project database.
        hybrid_activity_keys: Project keys for copied premise activities.
        early_input_bindings: Declared suppliers indexed by native parent and input.
    """

    background_scenario: str | None
    year: int
    provider_database: str
    source: _ActivityResolver
    ordinary: _ActivityResolver
    technology: _ActivityResolver
    biosphere: _FlowResolver
    catalogue: dict[str, object]
    material_recipe: EcoinventMaterialRecipe
    method_authorities: dict[str, dict[str, object]]
    complete_source_ids: dict[str, tuple[str, ...]]
    semantic_rows: list[dict[str, object]]
    technology_rows: list[dict[str, object]]
    traced_services: set[str]
    native_exchanges: dict[tuple[str, str], tuple[Any, ...]] = field(default_factory=dict)
    input_nodes: dict[tuple[str, str], Any] = field(default_factory=dict)
    hybrid_datasets: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)
    hybrid_activity_keys: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    early_input_bindings: dict[
        tuple[tuple[str, str, str, str], tuple[str, str, str, str]], dict[str, object]
    ] = field(default_factory=dict, init=False)

    def exchanges(self, activity: Any) -> tuple[Any, ...]:
        """Cache source exchanges and check transformed premise amounts.

        Exchange amounts must be finite; negative values are allowed.
        Unspecified uncertainty parameters retain their native markers.

        Raises:
            ValueError: If a transformed premise exchange amount is not finite.
        """
        key = cast(tuple[str, str], activity.key)
        if key not in self.native_exchanges:
            exchanges = tuple(activity.exchanges())
            if key[0] == self.technology.database_name and key[0] != self.source.database_name:
                for exchange in exchanges:
                    amount = float(exchange["amount"])
                    if not math.isfinite(amount):
                        raise ValueError(
                            f"Premise activity {key!r} has nonfinite amount {amount!r} "
                            f"for input {exchange['input']!r}. All exchange amounts must be "
                            "finite; prepare a valid source database in a new workspace."
                        )
            self.native_exchanges[key] = exchanges
        return self.native_exchanges[key]

    def input(self, exchange: Any) -> Any:
        """Resolve an exchange input once from the prepared database indexes."""

        key = cast(tuple[str, str], exchange["input"])
        if key not in self.input_nodes:
            if key[0] == self.ordinary.database_name:
                node = self.ordinary.node(key)
            elif key[0] == self.technology.database_name:
                node = self.technology.node(key)
            elif key[0] == self.biosphere.database_name:
                node = self.biosphere.node(key)
            elif key[0] == self.source.database_name:
                node = self.source.node(key)
            else:
                node = exchange.input
            self.input_nodes[key] = node
        return self.input_nodes[key]

    def exchange_json(self, exchanges: object) -> str:
        """Serialize the admitted native exchanges."""

        return json.dumps(
            [_exchange_record(exchange, self.input(exchange)) for exchange in cast(Any, exchanges)],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def resolver(self, source_role: str) -> _ActivityResolver:
        """Select the existing resolver for the declared provider source role."""
        return self.technology if source_role == "premise_only_provider" else self.ordinary

    def _hybrid_clone_key(self, activity: Any) -> tuple[str, str]:
        """Build the key for one early-year provider-technology clone."""
        source_key = cast(tuple[str, str], activity.key)
        digest = sha256(f"{source_key[0]}\0{source_key[1]}".encode()).hexdigest()[:24]
        return self.provider_database, f"hybrid_provider_technology__{digest}"

    def _hybrid_input_key(
        self,
        activity: Any,
        *,
        service_id: str,
        parent_activity: Any,
        source_amount: float,
    ) -> tuple[str, str]:
        """Use the declared supplier for this native parent and input pair.

        The parent distinguishes source duties merged into one premise market,
        such as the different heat supplies for PEI and silica production.
        """
        if not self.early_input_bindings:
            semantic = cast(dict[str, object], self.catalogue["semantic_contract"])
            authority = cast(dict[str, object], semantic["hybrid_provider_temporal_authority"])
            for process in cast(list[dict[str, object]], authority["processes"]):
                parent = cast(
                    tuple[str, str, str, str], tuple(cast(list[str], process["descriptor"]))
                )
                for binding in cast(list[dict[str, object]], process["inputs"]):
                    source = cast(
                        tuple[str, str, str, str], tuple(cast(list[str], binding["source"]))
                    )
                    self.early_input_bindings[parent, source] = binding
        binding = self.early_input_bindings[
            _input_semantics(parent_activity), _input_semantics(activity)
        ]
        if binding["source_role"] == "premise_only_provider":
            return self._hybrid_clone_activity(activity, service_id=service_id)
        ordinary = self.ordinary.activity(
            cast(dict[str, object], binding["selector"]), year=self.year
        )
        target_key = cast(tuple[str, str], ordinary.key)
        self.technology_rows.append(
            {
                "background_scenario": self.background_scenario,
                "anchor_year": self.year,
                "provider_service_id": service_id,
                "ledger_entry_type": "early_provider_ordinary_upstream_rebinding",
                "provider_source_role": "premise_only_provider",
                "provider_source_database": self.technology.database_name,
                "provider_source_code": str(activity["code"]),
                "provider_source_name": str(activity.get("name", "")),
                "provider_source_reference_product": str(activity.get("reference product", "")),
                "provider_source_unit": str(activity.get("unit", "")),
                "provider_source_location": str(activity.get("location", "")),
                "provider_parent_database": str(parent_activity.key[0]),
                "provider_parent_code": str(parent_activity.key[1]),
                "ordinary_provider_database": str(target_key[0]),
                "ordinary_provider_code": str(target_key[1]),
                "ordinary_provider_name": str(ordinary.get("name", "")),
                "ordinary_provider_reference_product": str(ordinary.get("reference product", "")),
                "ordinary_provider_unit": str(ordinary.get("unit", "")),
                "ordinary_provider_location": str(ordinary.get("location", "")),
                "source_exchange_amount": source_amount,
                "amount_sign_uncertainty_allocation_preserved": True,
                "provider_anchor_year": self.year,
                "general_background_database": self.ordinary.database_name,
                "general_background_anchor_year": 2025,
                "general_background_role": "ordinary_ecoinvent",
                "selected_background_member": self.background_scenario,
            }
        )
        return target_key

    def _hybrid_clone_activity(
        self,
        activity: Any,
        *,
        service_id: str,
    ) -> tuple[str, str]:
        """Copy a premise activity while replacing its ordinary background inputs.

        Args:
            activity: Premise activity to copy into the project provider database.
            service_id: Provider service that uses the activity.

        Returns:
            Brightway database and code of the project activity. Repeated requests for
            the same premise activity return the existing project key.

        Notes:
            Production points to the project copy. The declared technology graph is
            copied with its fixed ordinary ecoinvent 3.12 suppliers. Amounts, units,
            and exchange metadata are unchanged.
        """

        source_key = cast(tuple[str, str], activity.key)
        existing = self.hybrid_activity_keys.get(source_key)
        if existing is not None:
            return existing

        clone_key = self._hybrid_clone_key(activity)
        self.hybrid_activity_keys[source_key] = clone_key
        source_record = dict(activity)
        dataset = {
            key: value
            for key, value in source_record.items()
            if key not in {"database", "code", "id"}
        }
        dataset.update(
            {
                "database": clone_key[0],
                "code": clone_key[1],
                "type": str(activity.get("type", "process")),
                "frontier_service": True,
                "foreground_service": False,
                "hybrid_provider_technology_clone": True,
                "provider_source_database": self.technology.database_name,
                "provider_source_code": str(activity["code"]),
                "provider_source_role": "premise_only_provider",
                "provider_anchor_year": self.year,
                "general_background_database": self.ordinary.database_name,
                "general_background_anchor_year": 2025,
                "general_background_role": "ordinary_ecoinvent",
                "selected_background_member": self.background_scenario,
                "exchanges": [],
            }
        )
        self.hybrid_datasets[clone_key] = dataset

        generated_exchanges: list[dict[str, object]] = []
        for exchange in self.exchanges(activity):
            copied = _copied_native_exchange(exchange)
            exchange_type = str(exchange["type"])
            if exchange_type == "production":
                copied["input"] = clone_key
            elif exchange_type == "technosphere":
                input_node = self.input(exchange)
                if str(input_node.key[0]) == self.technology.database_name:
                    copied["input"] = self._hybrid_input_key(
                        input_node,
                        service_id=service_id,
                        parent_activity=activity,
                        source_amount=float(exchange["amount"]),
                    )
            generated_exchanges.append(copied)
        dataset["exchanges"] = generated_exchanges
        self.technology_rows.append(
            {
                "background_scenario": self.background_scenario,
                "anchor_year": self.year,
                "provider_service_id": service_id,
                "ledger_entry_type": "early_premise_provider_technology_clone",
                "provider_source_role": "premise_only_provider",
                "provider_source_database": self.technology.database_name,
                "provider_source_code": str(activity["code"]),
                "provider_source_name": str(activity.get("name", "")),
                "provider_source_reference_product": str(activity.get("reference product", "")),
                "provider_source_unit": str(activity.get("unit", "")),
                "provider_source_location": str(activity.get("location", "")),
                "frontier_clone_database": clone_key[0],
                "frontier_clone_code": clone_key[1],
                "provider_anchor_year": self.year,
                "general_background_database": self.ordinary.database_name,
                "general_background_anchor_year": 2025,
                "general_background_role": "ordinary_ecoinvent",
                "selected_background_member": self.background_scenario,
            }
        )
        return clone_key

    def isolate_early_provider_upstream(self, dataset: dict[str, object]) -> dict[str, object]:
        """Record provider and general-background attribution after recipe binding.

        Early premise inputs already point to project technology copies with
        their declared ecoinvent 2025 suppliers. Retain those exchanges and
        identify their technology source in the returned metadata. When both
        resolvers use the same database, record its matching member and year.
        """

        if self.ordinary.database_name == self.technology.database_name:
            return {
                **dataset,
                "provider_source_database": self.technology.database_name,
                "provider_source_role": "matching_member_premise_provider",
                "provider_anchor_year": self.year,
                "general_background_database": self.ordinary.database_name,
                "general_background_anchor_year": self.year,
                "general_background_role": "matching_member_premise_general_background",
                "selected_background_member": self.background_scenario,
            }
        used_premise_technology = dataset.get(
            "provider_source_database"
        ) == self.technology.database_name or any(
            str(raw["type"]) == "technosphere"
            and cast(tuple[str, str], raw["input"]) in self.hybrid_datasets
            for raw in cast(list[dict[str, object]], dataset["exchanges"])
        )
        return {
            **dataset,
            "provider_source_database": (
                self.technology.database_name
                if used_premise_technology
                else self.ordinary.database_name
            ),
            "provider_source_role": (
                "premise_only_provider" if used_premise_technology else "native_ecoinvent_provider"
            ),
            "provider_anchor_year": self.year,
            "general_background_database": self.ordinary.database_name,
            "general_background_anchor_year": 2025,
            "general_background_role": "ordinary_ecoinvent",
            "selected_background_member": self.background_scenario,
        }

    def activity(
        self,
        selector: dict[str, object],
        *,
        source_role: str,
        service_id: str,
        component_id: str,
        semantic_class: str,
    ) -> tuple[Any, dict[str, object]]:
        """Resolve one exact source activity for the current recipe."""
        resolver = self.resolver(source_role)
        activity = resolver.activity(selector, year=self.year)
        native_exchanges = tuple(
            exchange
            for exchange in self.exchanges(activity)
            if str(exchange["type"]) != "production"
        )
        row = {
            "background_scenario": self.background_scenario,
            "anchor_year": self.year,
            "ledger_entry_type": "native_component",
            "semantic_class": semantic_class,
            "provider_service_id": service_id,
            "component_id": component_id,
            "source_class": source_role,
            "provider_source_role": source_role,
            "provider_source_database": resolver.database_name,
            "provider_anchor_year": self.year,
            "general_background_database": self.ordinary.database_name,
            "general_background_anchor_year": (
                2025 if self.ordinary.database_name != self.technology.database_name else self.year
            ),
            "general_background_role": (
                "ordinary_ecoinvent"
                if self.background_scenario is None
                or self.ordinary.database_name != self.technology.database_name
                else "matching_member_premise_general_background"
            ),
            "selected_background_member": self.background_scenario,
            "native_database_name": resolver.database_name,
            "activity_code": str(activity["code"]),
            "activity_name": str(activity.get("name", "")),
            "reference_product": str(activity.get("reference product", "")),
            "native_unit": str(activity.get("unit", "")),
            "location": str(activity.get("location", "")),
            "retained_technosphere_json": self.exchange_json(
                exchange for exchange in native_exchanges if str(exchange["type"]) == "technosphere"
            ),
            "removed_technosphere_json": "[]",
            "retained_biosphere_json": self.exchange_json(
                exchange for exchange in native_exchanges if str(exchange["type"]) == "biosphere"
            ),
            "project_terms_json": "[]",
        }
        self.technology_rows.append(row)
        return activity, row


def _copied_native_exchange(exchange: Any) -> dict[str, object]:
    """Copy one native exchange into a package-owned project activity."""

    record = dict(exchange)
    record.pop("output", None)
    record["input"] = cast(tuple[str, str], exchange["input"])
    record["amount"] = float(exchange["amount"])
    record["type"] = str(exchange["type"])
    return record


def _scaled_native_exchange(exchange: Any, multiplier: float) -> dict[str, object]:
    """Scale an admitted deterministic/lognormal supply and its material bounds.

    Selected water, oxide and treatment suppliers use these distributions.
    A positive multiplier shifts a lognormal location and retains its logarithmic spread;
    deterministic locations and material bounds scale
    directly. Uncertainty is represented separately for each native row.
    """
    record = _copied_native_exchange(exchange)
    record["amount"] = float(exchange["amount"]) * multiplier
    if "loc" in exchange:
        record["loc"] = (
            float(exchange["loc"]) + math.log(multiplier)
            if exchange.get("uncertainty type", 0) == 2
            else float(exchange["loc"]) * multiplier
        )
    for bound in ("minimum", "maximum"):
        if bound in exchange:
            record[bound] = float(exchange[bound]) * multiplier
    return record
