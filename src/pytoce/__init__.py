"""Public pyToCE interface.

Public workflow functions are imported when their attributes are first used.
"""

import sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"

_PUBLIC_EXPORT_MODULES = {
    "download_data": "pytoce.download_data.download_data",
    "prepare_backgrounds": "pytoce.prepare_backgrounds.prepare_backgrounds",
    "run_asr": "pytoce.run_asr.run_asr",
    "run_lca": "pytoce.run_lca.run_lca",
    "run_mfa": "pytoce.run_mfa.run_mfa",
    "run_optimization": "pytoce.run_optimization.run_optimization",
    "set_workspace": "pytoce.set_workspace.set_workspace",
}

if TYPE_CHECKING:
    from .download_data.download_data import download_data
    from .prepare_backgrounds.prepare_backgrounds import prepare_backgrounds
    from .run_asr.run_asr import run_asr
    from .run_lca.run_lca import run_lca
    from .run_mfa.run_mfa import run_mfa
    from .run_optimization.run_optimization import run_optimization
    from .set_workspace.set_workspace import set_workspace

__all__ = [
    "set_workspace",
    "download_data",
    "prepare_backgrounds",
    "run_mfa",
    "run_lca",
    "run_asr",
    "run_optimization",
]


def _load_public_workflow(name: str) -> Any:
    """Load one public workflow and preserve root callable attributes."""

    module_name = _PUBLIC_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'pytoce' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """Load one implemented package-level workflow when first accessed."""

    return _load_public_workflow(name)


def __dir__() -> list[str]:
    """Return discoverable namespace names, including the lazy workflows."""

    return sorted(set(globals()) | set(__all__))


class _PublicInterfaceModule(ModuleType):
    """Preserve lazy root callables when subpackages were imported first."""

    def __getattribute__(self, name: str) -> Any:
        """Return a package attribute after applying lazy export resolution."""
        value = super().__getattribute__(name)
        if name in _PUBLIC_EXPORT_MODULES and isinstance(value, ModuleType):
            return _load_public_workflow(name)
        return value


sys.modules[__name__].__class__ = _PublicInterfaceModule
