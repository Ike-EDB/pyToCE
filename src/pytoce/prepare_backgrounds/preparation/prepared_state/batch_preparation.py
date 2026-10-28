"""Prepare the selected background scenarios."""

import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ..premise import (
    BackgroundScenario,
    PremiseIAMInput,
    acquire_iam_input,
)
from ..providers.native_resolution import _NativeResolverCache
from .member_preparation import PreparedBackgroundResult, prepare_member
from .shared_preparation import SharedAuthority, ensure_shared_authority
from .source_authority import Sources


def write_progress(background_scenario: str, phase: str) -> None:
    """Print one background preparation step."""
    message = {
        "reuse": "reusing the completed background scenario",
        "premise_update": "preparing the premise background databases",
        "provider_services": "preparing the plastics provider databases",
    }[phase]
    print(
        f"prepare_backgrounds: {background_scenario}: {message}",
        file=sys.stderr,
    )


def prepare_members(
    workspace: Path,
    sources: Sources,
    shared: SharedAuthority,
    scenarios: tuple[BackgroundScenario, ...],
    iam_inputs: dict[str, PremiseIAMInput],
    *,
    years: tuple[int, ...],
    premise_key: str | None,
    progress: Callable[[str, str], None],
) -> tuple[PreparedBackgroundResult, ...]:
    """Prepare the selected background scenarios using the shared data."""

    provider = replace(
        sources.provider,
        routes=pd.DataFrame(shared.provider_routes),
    )
    provider_resolvers = _NativeResolverCache(
        shared.names.source_database,
        shared.names.biosphere_database,
    )
    return tuple(
        prepare_member(
            workspace,
            sources,
            shared,
            provider,
            iam_inputs[scenario.background_scenario],
            scenario,
            years=years,
            premise_key=premise_key,
            progress=progress,
            provider_resolvers=provider_resolvers,
        )
        for scenario in scenarios
    )


def prepare_selected(
    workspace: Path,
    sources: Sources,
    scenarios: tuple[BackgroundScenario, ...],
    *,
    years: tuple[int, ...],
    premise_key: str | None,
) -> tuple[PreparedBackgroundResult, ...]:
    """Prepare the requested background scenarios and years."""

    iam_inputs = {
        scenario.background_scenario: acquire_iam_input(
            workspace,
            scenario,
            premise_authority=sources.premise_authority,
        )
        for scenario in scenarios
    }
    shared = ensure_shared_authority(workspace, sources)
    return prepare_members(
        workspace,
        sources,
        shared,
        scenarios,
        iam_inputs,
        years=years,
        premise_key=premise_key,
        progress=write_progress,
    )
