# Package Architecture Notes

Audience: developers contributing changes to package internals.

These notes explain how the source tree is organized, which module owns which
responsibility, and where the corresponding package tests live.

## Public Surface

The stable public Python API is exposed from
[`src/pytoce/__init__.py`](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/__init__.py).
The package root resolves each public function lazily when its attribute is
first accessed. Other `src/pytoce/**/__init__.py` files are package markers.
Internal imports target the concrete modules that own each responsibility.

Workspace creation, downloads, Brightway operations, and calculations start
through their corresponding workflow functions.

## Source Architecture Notes

Architecture notes live alongside the package source tree:

| Scope | Architecture note | Focused unit tests |
| --- | --- | --- |
| Package dependencies and imports | [Package architecture](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/ARCHITECTURE.md) | `tests/unit/test_public_imports.py` |
| Workspace and installed resources | [Workspace](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/set_workspace/ARCHITECTURE_set_workspace.md) | `tests/unit/set_workspace/` |
| Data downloads, licensed inputs, and material recipe | [Data downloads](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/download_data/ARCHITECTURE_download_data.md) | `tests/unit/download_data/` |
| Background databases and provider inventories | [Background preparation](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/prepare_backgrounds/ARCHITECTURE_prepare_backgrounds.md) | `tests/unit/prepare_backgrounds/` |
| Scenarios, cohorts, material flows, and MFA results | [MFA](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/run_mfa/ARCHITECTURE_run_mfa.md) and [REMIND-MFA inputs](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/run_mfa/remind_mfa/ARCHITECTURE_remind_mfa.md) | `tests/unit/run_mfa/` |
| Foreground inventories, characterization, and contributions | [LCA](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/run_lca/ARCHITECTURE_run_lca.md) | `tests/unit/run_lca/` |
| Monitoring ASRs and shared assessment calculations | [ASR](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/run_asr/ARCHITECTURE_run_asr.md) | `tests/unit/run_asr/` |
| Allocated carrying capacity inputs, candidate search, and selection | [Optimization](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/run_optimization/ARCHITECTURE_run_optimization.md) | `tests/unit/run_optimization/` |

## Responsibility Boundary

- `set_workspace` owns workspace selection and installed input resources.
- `download_data` owns public source downloads, licensed archive checks and
  the locally derived plastics recipes.
- `prepare_backgrounds` owns LCIA definitions, provider catalogs, Brightway
  projects, premise members, and prepared background results.
- `run_mfa` owns scenario compilation, REMIND-MFA inputs, correction, material flow
  calculation, and MFA results.
- `run_lca` owns foreground construction, time-explicit inventories,
  characterization, contributions, and LCA results.
- `run_asr` owns monitoring requests and annual and cumulative ASR tables.
- `run_optimization` owns allocated carrying capacity (aCC) inputs,
  response tables, evaluation of
  intervention combinations, selection, reporting, the cache for completed
  intervention calculations, and optimization results.
- `shared.runtime` contains serialization, filesystem, workspace, and
  workspace-guide code used by more than one workflow.
- `shared.assessment` owns the annual and cumulative ASR calculation used by
  monitoring and target setting.

Dependencies point from the seven public functions to the modules that perform
each calculation. `run_lca` reads completed MFA and prepared background
results. `run_asr` reads completed LCA results and reuses the existing aCC input
owner under `run_optimization/input`. `run_optimization` reads completed MFA and required baseline LCA results
for each requested foreground scenario. It evaluates new intervention candidates
through the shared MFA owner and reuses matching completed candidates. Data
downloads, background preparation and prerequisite MFA/LCA calculations remain
separate workflows.

## Scope Identity And Reuse

The manifest is written after all result files are complete. A completed
manifest enables fast reuse. Refresh applies to the selected workflow's results.
Calculate a new result when an existing product is damaged or inconsistent.

## Testing and Quality Checks

Ruff is the formatting and linting baseline. Pyright is the static type checker
for source code.

Select the tests for the changed owner and its direct consumers, including
relevant workflow boundaries in `tests/integration/`. Public documentation checks
live in `tests/docs/`; distribution, metadata, and public docstring checks live
in `tests/package/`.

Follow the [contributor guide](contributing.md) for Python 3.11 and 3.12 support,
Ruff, Pyright, focused tests, and release validation. Scientific imports and tests
must use disposable Brightway storage. Documentation and distribution builds check rendering and packaging.
Scientific validation checks the methods and calculated results.
