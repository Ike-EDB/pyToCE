# pyToCE

![pyToCE logo](https://github.com/Ike-EDB/pyToCE/raw/refs/heads/main/docs/_static/pytoce-logo.svg)

**Documentation:** [pytoce.readthedocs.io](https://pytoce.readthedocs.io/)
reproduces the documentation of this GitHub repository.

**Issues and feedback:** Please report bugs, unexpected behavior, documentation gaps,
and installation or workflow issues on the [GitHub Issues page](https://github.com/Ike-EDB/pyToCE/issues).

**Theory of Change for circular economy.**

pyToCE is a Python package that operationalizes a Theory of Change-based
framework linking inherent circularity performance to absolute environmental
sustainability. Dynamic material flow analysis (MFA) translates circular economy
(CE) performance levels into annual material-flow trajectories. Time-explicit
prospective life cycle assessment (LCA) then quantifies the associated
environmental burdens. Finally, absolute environmental sustainability assessment
(AESA) compares these burdens with allocated carrying capacities. These can be
calculated with the [pyaesa Python package](https://github.com/AESAtoolkit/pyaesa)
through the optional tutorial or supplied by the user.

## Package Scope

The current version of `pyToCE` covers only the plastics sector. It models the
period from 2019 to 2100, with the world divided into 12 regions and plastics
represented by 12 distinct families distributed among seven end-use groups.
It builds on [REMIND-MFA 0.2.0](https://github.com/pik-piam/remind-mfa) and its
[input data 1.0.0](https://doi.org/10.5281/zenodo.18019915) to model stocks,
demand, virgin production, virgin feedstock composition, trade, collection,
recycling, incineration, landfill, open burning, and open dumping.

The LCA uses [ecoinvent 3.12](https://ecoinvent.org/database/) Allocation, cut-off
by classification in [Brightway](https://github.com/brightway-lca).
Prospective background inventories are generated with
[premise](https://github.com/polca/premise) and linked to the year of each life
cycle activity by [bw_timex](https://github.com/brightway-lca/bw_timex).

pyToCE can use the four REMIND SSP2 background scenarios available in
[premise 2.4.6](https://github.com/polca/premise),
covering climate policy pathways ranging from below 1.5 °C to above 3 °C of
warming by 2100.

Environmental burdens are characterized using the 16 EF 3.1 midpoint impact
categories and/or 10 PB-LCIA impact categories.

## Workflow overview

The figure below summarizes the main package data sources, public workflow
functions, study objectives, and output families.

[![pyToCE workflow: material flow analysis, life cycle assessment, allocated carrying capacities, sustainability assessment, and target setting](https://github.com/Ike-EDB/pyToCE/raw/refs/heads/main/docs/_static/workflow.svg)](https://github.com/Ike-EDB/pyToCE/raw/refs/heads/main/docs/_static/workflow.svg)

## Study objectives and recommended routes

The framework can be used in two directions: monitoring and target setting. For
monitoring, CE performance levels are first set, and the resulting environmental
burdens are calculated and compared with allocated carrying capacities. For
target setting, allocated carrying capacities are specified first, and the package
searches for combinations of CE performance levels whose environmental burdens
could meet these requirements. CE performance levels are therefore an input to
monitoring and an output of target setting.
A noncompliant fallback is retained when no tested combination complies.

## Installation

Python 3.11 or 3.12 is required.

On macOS with Apple Silicon, install the LCA solver build requirements using
[Homebrew](https://brew.sh/):

```console
brew install suite-sparse swig pkg-config
```

Install the package from PyPI:

```console
python -m pip install pytoce
```

## ecoinvent License and Data

Before `download_data`, place these licensed ecoinvent 3.12 archives in
`data_raw/ecoinvent/`:

```text
ecoinvent_3.12_cutoff_ecoSpold02.7z
ecoinvent 3.12_LCIA_implementation.7z
```

pyToCE verifies and registers these locally supplied archives. The user must
hold the required ecoinvent license.

## Public Workflow Function Map

The package root exports seven functions. `set_workspace` selects the active
workspace for the current Python session. The other functions use that workspace,
print concise progress, and return reports with the result paths and status.

| Function | Required inputs or completed results | What it computes or prepares and writes |
| --- | --- | --- |
| `set_workspace` | A workspace path. | Workspace directories, templates, schemas, and methodological resources. |
| `download_data` | An active workspace and the licensed ecoinvent archives. | Public source files, registered licensed inputs, and the material recipe. |
| `prepare_backgrounds` | Completed `download_data` results and the selected REMIND SSP2 backgrounds. | Brightway background databases, LCIA methods, and dated provider inventories. |
| `run_mfa` | Completed `download_data` results and a foreground scenario. | Annual plastics stocks, cohorts, material flows, and treatment results. |
| `run_lca` | Completed MFA results and prepared backgrounds. | Annual and period-total environmental burdens and contribution tables. |
| `run_asr` | Completed LCA results and a complete allocated carrying capacity result. | Annual and cumulative ASRs for user-defined CE performance trajectories. |
| `run_optimization` | A complete allocated carrying capacity result, foreground MFA, prepared backgrounds, and required baseline LCA results. | Selected CE performance levels, annual trajectories, environmental burdens, and absolute sustainability ratios. |

Run the prerequisite workflows before their consumers.

## Quick start

Run `set_workspace(...)` once at the beginning of each Python session.

Workspace setup copies templates, schemas, the LCI workbook, article and
Methodological appendix PDFs, and citation guidance.

```python
from pathlib import Path

from pytoce import set_workspace

report = set_workspace(Path("pytoce_workspace"))
print(report.as_dict())
```

Runnable foreground scenarios are stored in
`data_raw/foreground_scenarios/scenarios/`. Allocated carrying capacity (aCC)
results can be calculated following the
[optional pyaesa tutorial](https://github.com/Ike-EDB/pyToCE/blob/main/tutorials/optional_workflows/pyaesa_allocated_carrying_capacity.ipynb)
or supplied by the user. Store them in `data_raw/allocated_cc/results/`.
The installed templates describe the v1 input schemas.

The example uses 2019frozen_SSP2demand, whose executable identifier is
`frozen_2019_default_baseline`. It retains default SSP2 stock demand and lifetimes
while collection, treatment and virgin feedstock shares stay at their 2019 values.
New backgrounds require a premise decryption key. Request it as described in the
[premise documentation](https://premise.readthedocs.io/en/latest/getting_started/preparation.html).
Completed backgrounds can be reused. After placing the ecoinvent archives in
`data_raw/ecoinvent/`, run the common prerequisite calculations:

```python
from getpass import getpass

from pytoce import (
    download_data,
    prepare_backgrounds,
    run_asr,
    run_lca,
    run_mfa,
    run_optimization,
)

download_data()
prepare_backgrounds(premise_key=getpass("premise decryption key: "))
foreground = "frozen_2019_default_baseline"
run_mfa(foreground_scenario=foreground)
run_lca(foreground_scenario=foreground)
```

After the LCA, prepare or supply a complete aCC result and copy it into
`data_raw/allocated_cc/results/`. The frozen foreground's LCA results supply the
baseline for selected-result reporting. Calculate `remind_mfa_default`
separately if you also want to compare the two foregrounds. Replace the aCC
result name below before running. For monitoring, assess the example
foreground with:

```python
run_asr(
    foreground_scenario=foreground,
    allocated_cc_result="name_of_your_allocated_cc_result",
)
```

For target setting, search for CE performance target levels with:

```python
run_optimization(
    foreground_scenario="frozen_2019_default_baseline",
    allocated_cc_result="name_of_your_allocated_cc_result",
    candidate_workers=1,
    initial_grid_step_pct=20,
    endpoint_refinement_steps_pct=(10, 5),
)
```

The [Workflow Reference](https://pytoce.readthedocs.io/en/latest/workflow_reference.html) explains background
selection, runtime controls, search settings, and reuse of completed calculations.

## Set of tutorials

The README is the tutorial navigator.
Follow core Tutorials 1--4, then choose Tutorial 5m for monitoring or
Tutorial 5o for target setting. Each notebook selects the workspace, so it
can be opened independently once its required inputs and parent results
are complete. The optional pyaesa notebook follows the 2019 LCA and supplies
allocated carrying capacities for either study objective.

### Core prerequisites tutorials:

| Key | Notebook |
| --- | --- |
| Workspace and source data | [`tutorials/core_workflow/1_download_data.ipynb`](https://github.com/Ike-EDB/pyToCE/blob/main/tutorials/core_workflow/1_download_data.ipynb) |
| Backgrounds | [`tutorials/core_workflow/2_prepare_backgrounds.ipynb`](https://github.com/Ike-EDB/pyToCE/blob/main/tutorials/core_workflow/2_prepare_backgrounds.ipynb) |
| Plastics MFA | [`tutorials/core_workflow/3_run_mfa.ipynb`](https://github.com/Ike-EDB/pyToCE/blob/main/tutorials/core_workflow/3_run_mfa.ipynb) |
| Time-explicit LCA | [`tutorials/core_workflow/4_run_lca.ipynb`](https://github.com/Ike-EDB/pyToCE/blob/main/tutorials/core_workflow/4_run_lca.ipynb) |

### Study objectives tutorials:

| Key | Notebook |
| --- | --- |
| 5m. Monitoring ASRs | [`tutorials/core_workflow/5m_run_asr.ipynb`](https://github.com/Ike-EDB/pyToCE/blob/main/tutorials/core_workflow/5m_run_asr.ipynb) |
| 5o. Optimization | [`tutorials/core_workflow/5o_run_optimization.ipynb`](https://github.com/Ike-EDB/pyToCE/blob/main/tutorials/core_workflow/5o_run_optimization.ipynb) |

### Optional tutorials:

| Key | Notebook |
| --- | --- |
| Optional pyaesa aCC calculation | [`tutorials/optional_workflows/pyaesa_allocated_carrying_capacity.ipynb`](https://github.com/Ike-EDB/pyToCE/blob/main/tutorials/optional_workflows/pyaesa_allocated_carrying_capacity.ipynb) |

### Methodological notes:

The article provides the methodological basis for `pyToCE`. The following
methodological resources supporting the package implementation are distributed
with `pyToCE`:

| Location | Notes |
| --- | --- |
| [Article](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/set_workspace/data/methodological_notes/pytoce_article.pdf) | Article providing the methodological basis for `pyToCE`. |
| [Methodological appendix](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/set_workspace/data/methodological_notes/pytoce_methodological_appendix.pdf) | Detailed methodological description of the framework and its use for monitoring and target setting. |
| [LCI workbook](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/set_workspace/data/methodological_notes/LCI_per_unit.xlsx) | Per-unit inventories, quantities, and sources used for polymer production, conversion, and end-of-life treatment. |
| [Recommended citations](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/set_workspace/data/methodological_notes/recommended_citations.txt) | Software, data, and method references used by the package. |

`set_workspace(...)` copies these files into the active workspace under
`data_raw/methodological_notes/`.

## Documentation

Use the [API Reference](https://pytoce.readthedocs.io/en/latest/api.html) for exact
signatures and parameter contracts. Use [Tutorials](https://pytoce.readthedocs.io/en/latest/tutorial.html)
for the notebook index.

- [Package Overview](https://pytoce.readthedocs.io/en/latest/overview.html)
- [Workflow Reference](https://pytoce.readthedocs.io/en/latest/workflow_reference.html)
- [Methodological Notes](https://pytoce.readthedocs.io/en/latest/publications_and_resources.html)

For development and documentation build instructions, see
[CONTRIBUTING.md](https://github.com/Ike-EDB/pyToCE/blob/main/CONTRIBUTING.md).

## Citation

Please cite the article when using `pyToCE`, as it constitutes the
methodological basis for `pyToCE`.

Use [CITATION.cff](https://github.com/Ike-EDB/pyToCE/blob/main/CITATION.cff) for the software citation.

For data source citations, see the
[recommended citation guide](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/set_workspace/data/methodological_notes/recommended_citations.txt).

## License

The independently authored package code is distributed under the BSD 3-Clause
License. Upstream software, data, and licensed archives retain their own terms.
See [LICENSE](https://github.com/Ike-EDB/pyToCE/blob/main/LICENSE) and [NOTICE](https://github.com/Ike-EDB/pyToCE/blob/main/NOTICE).

## Contributing

To propose direct code modifications, see [`CONTRIBUTING.md`](https://github.com/Ike-EDB/pyToCE/blob/main/CONTRIBUTING.md).
Contributors follow the [Code of Conduct](https://github.com/Ike-EDB/pyToCE/blob/main/CODE_OF_CONDUCT.md).
