"""Read completed MFA products for LCA and optimization calculations."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import pandas as pd

from .mfa import MFAEvaluation

_MFA_INPUT_TABLES = {
    "model_flows": "flows.parquet",
    "derived_outputs": "derived_annual_outputs.parquet",
    "material_realization": "material_realization.parquet",
    "nominal_feedstock_equivalents": "nominal_feedstock_equivalents.parquet",
    "cohort_state": "cohort_state.parquet",
    "cohort_material_identity": "cohort_material_identity.parquet",
}


@dataclass(frozen=True, slots=True)
class MFAResultReference:
    """A completed manifest and its byte identity for new descendants."""

    path: Path
    manifest: dict[str, object]
    sha256: str


def read_mfa_result_manifest(path: Path) -> MFAResultReference:
    """Read the manifest and record its digest for downstream result identities."""
    content = path.read_bytes()
    return MFAResultReference(path, json.loads(content), sha256(content).hexdigest())


def read_mfa_result(reference: MFAResultReference) -> MFAEvaluation:
    """Read the six MFA tables and their recorded MFA state and material composition identities.

    Preserve the producer's values, units, axes, dtypes, ordering and sparse
    cohort representation. The MFA model owns balance and feasibility
    calculations.
    """
    tables = {
        role: pd.read_parquet(reference.path.parent / name)
        for role, name in _MFA_INPUT_TABLES.items()
    }
    scientific = cast(dict[str, str], reference.manifest["scientific_authorities"])
    return MFAEvaluation(
        **tables,
        physical_state_identity=scientific["physical_state_identity"],
        material_identity_method=scientific["material_identity_method"],
    )
