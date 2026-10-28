# Contributor Guide

Thank you for your interest in improving `pyToCE`.

`pyToCE` is an open-source Python package for target-driven circular economy
planning through absolute environmental sustainability assessment (AESA).
The project welcomes contributions in the form of bug reports, feature requests,
documentation improvements, tests, and pull requests.

## How To Set Up A Development Environment

Use Python 3.12 for development and intermediate checks. From the repository
root, create an environment and a disposable Brightway store.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,docs]"
$env:BRIGHTWAY2_DIR = python -c "import tempfile; print(tempfile.mkdtemp(prefix='pytoce-tests-'))"
$env:OMP_NUM_THREADS = $env:MKL_NUM_THREADS = $env:OPENBLAS_NUM_THREADS = $env:NUMEXPR_NUM_THREADS = "1"
python tests/provision_remind_mfa_source.py
python -m pytest -q -n 3 --dist=loadfile --max-worker-restart=0 --durations=25 tests/unit tests/integration
```

On Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
export BRIGHTWAY2_DIR="$(python -c 'import tempfile; print(tempfile.mkdtemp(prefix="pytoce-tests-"))')"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python tests/provision_remind_mfa_source.py
python -m pytest -q -n 3 --dist=loadfile --max-worker-restart=0 --durations=25 tests/unit tests/integration
```

The setup command downloads the public REMIND-MFA 0.2.0 source code and
configuration into `build/test_dependencies/remind_mfa`. It verifies the downloaded
archive once and reuses the provisioned directory. Tests read that source and
create small synthetic model inputs in temporary directories. If you already
have the exact source, set `REMIND_MFA_TEST_SOURCE` to its extracted root and
omit the setup command. Pytest uses that supplied source directly.
Pytest gives its controller and each worker fresh Brightway and temporary
directories before importing scientific fixtures. Tests from one file stay
together and share their module-scoped fixtures. Each worker uses one
numerical-library thread, and a worker failure stops its calculation. Keep
`BRIGHTWAY2_DIR` set to disposable storage before scientific imports in scripts
outside pytest too.

Do not commit licensed ecoinvent data, downloaded sources, prepared databases,
scientific results, credentials, machine-local paths, build artifacts, or
private research support.

## Scientific changes

Before changing a scientific method:

1. identify its source and current owner;
2. state whether inventories, stage years, providers, mass balance,
   characterization, contributions, or aggregation can change;
3. update the smallest meaningful deterministic test;
4. document the method, provenance, and limitations in the same change.

## Code changes

- Support Python 3.11 and 3.12, with type annotations and Google-style docstrings.
- Keep one owner for each responsibility and use direct, readable data flow.
- Preserve lazy package imports and the seven-function public API.
- Add no command-line interface.
- Test an invariant once at the owner that establishes it.

During development, select tests for the changed behavior and its direct
consumers. For example, a workspace installation change uses
`tests/unit/set_workspace/test_workspace.py`. Reuse accepted observations for
unchanged code when the source and execution conditions remain compatible.
Run the documentation and artifact commands when the change affects those outputs:

```console
python -m ruff check .
python -m ruff format --check .
python -m pyright src/pytoce
python -m pytest -q tests/unit/set_workspace/test_workspace.py
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
```

Expect the ordinary test suite to take roughly thirty minutes with coverage
disabled, excluding setup. Mark bounded scientific integration checks
and run them separately. The ordinary suite uses local source fixtures and
synthetic inputs.

The commands below establish a complete validation baseline for a new release
candidate on Python 3.12 and 3.11, after implementation and intermediate Python
3.12 checks are complete. For an already validated candidate, use focused
regression checks, exact coverage for changed source and compatible accepted
observations for unchanged source; repeat affected documentation, artifact and
installation checks. Broaden validation when an observed failure or a changed
dependency requires it.

Set `COVERAGE_FILE` to an absolute path in
fresh disposable storage. In the commands below, replace `<repository>` with
the absolute checkout path. Collect ordinary coverage in parallel, then append
the bounded scientific integrations and documentation/artifact checks serially:

```console
python -m pytest -q -n 3 --dist=loadfile --max-worker-restart=0 -m "not scientific_integration" tests/unit tests/integration --cov=pytoce --cov-branch --cov-config="<repository>/pyproject.toml" --cov-fail-under=0 --cov-report=term-missing
python -m pytest -q -n 0 -m scientific_integration tests/unit tests/integration --cov=pytoce --cov-branch --cov-append --cov-config="<repository>/pyproject.toml" --cov-fail-under=0 --cov-report=term-missing
python -m pytest -q -n 0 -m= tests/docs tests/package --cov=pytoce --cov-branch --cov-append --cov-config="<repository>/pyproject.toml" --cov-fail-under=0 --cov-report=term-missing
python -m coverage report
python -m coverage xml
```

The partial runs retain their reports even below the final threshold. Final
line and branch coverage must each be 100%, without exclusions hiding maintained
package code. Keep one `COVERAGE_FILE` across these commands; pytest-cov combines
the workers automatically. Do not enable workers globally or run the serial
groups alongside another scientific calculation. The documentation checks require
Poppler's `pdfinfo`, `pdftotext` and `pdftoppm` on `PATH`.

The Package Tests workflow runs through a manual dispatch and supports
private repositories. In **Actions > Package Tests > Run workflow**, select
the candidate branch and `test_scope`. Keep candidate commits and validation
artifacts in the private repository; publication is a separate repository-owner
action after final validation. Its ordinary suite uses one pytest worker to
share scientific fixture memory across tests on GitHub-hosted private runners.
Scientific integrations, documentation/artifact checks and native macOS checks
remain serial.

The workflow uses Python 3.12 for `lite`. To establish the full baseline using
hosted validation, select `full` and leave
`python_version` at `3.12`; after its exact coverage gate passes, run `full`
again with `python_version` set to `3.11`. Each invocation runs Windows/Linux
validation and native macOS solver checks for the selected version, then
combines its platform measurements and checks the line and branch totals
separately. Both invocations must pass for that baseline. `lite` uses Python 3.12.

For the full baseline, build and inspect wheel and source-distribution files
from a clean release candidate commit. Install each artifact in a fresh Python
3.11 and 3.12 environment before accepting that baseline. Later acceptance
combines this evidence with the required checks for changed behavior and artifacts.

## How To Submit Changes

Your pull request should include:

- a short summary of the change
- the reason for the change
- tests and checks run
- documentation updates, when relevant
- any scientific assumptions or limitations, when relevant
