---
myst:
  heading_anchors: 3
---

# Workflow Reference

This reference maps the main `pyToCE` workflows from data preparation to AESA
outputs. It identifies the data sources used by each phase, the prerequisite
steps to run, the public functions that produce each output, and the matching
tutorials.

The package root exports seven functions. `set_workspace` selects the active
workspace for the current Python session. The other functions use that workspace,
print concise progress, and return reports with the result paths and status.
The workflows provide reading guides explaining their tables, units and
limitations. Completed products have manifests recording their inputs and methods.

```text
set_workspace -> download_data -> run_mfa
download_data -> prepare_backgrounds
completed MFA + prepared background -> run_lca
completed LCA + allocated carrying capacity result -> run_asr
allocated carrying capacity result + foreground + prepared backgrounds + required baseline LCA results
    -> run_optimization
```

## Public Workflow Function Map

Install the package as described in {ref}`installation`.
The seven public functions have distinct prerequisites and outputs:

| Function | Required inputs or completed results | What it computes or prepares and writes |
| --- | --- | --- |
| {func}`~pytoce.set_workspace` | A workspace path. | Workspace directories, templates, schemas, and methodological resources. |
| {func}`~pytoce.download_data` | An active workspace and the licensed ecoinvent archives. | Public source files, registered licensed inputs, and the material recipe. |
| {func}`~pytoce.prepare_backgrounds` | Completed `download_data` results and the selected REMIND SSP2 backgrounds. | Brightway background databases, LCIA methods, and dated provider inventories. |
| {func}`~pytoce.run_mfa` | Completed `download_data` results and a foreground scenario. | Annual plastics stocks, cohorts, material flows, and treatment results. |
| {func}`~pytoce.run_lca` | Completed MFA results and prepared backgrounds. | Annual and period-total environmental burdens and contribution tables. |
| {func}`~pytoce.run_asr` | Completed LCA results and a complete allocated carrying capacity result. | Annual and cumulative ASRs for user-defined CE performance trajectories. |
| {func}`~pytoce.run_optimization` | A complete allocated carrying capacity result, foreground MFA, prepared backgrounds, and required baseline LCA results. | Selected CE performance levels, annual trajectories, environmental burdens, and absolute sustainability ratios. |

Run the prerequisite workflows before their consumers.
Use the {doc}`API reference <api>` for signatures and parameter
contracts. Use the {doc}`tutorial page <tutorial>` for the notebook index and
complete tutorial content.

### Choose the study objective

Monitoring starts from user-defined CE performance trajectories and follows the
causal chain from CE performance to material flows, environmental burdens,
allocated carrying capacities, and finally absolute environmental sustainability.
You can use a packaged foreground scenario or define your own.
Run `set_workspace`, `download_data`, `prepare_backgrounds`, `run_mfa`, and
`run_lca`, supply the allocated carrying capacities, then call `run_asr`.

Target setting starts from the allocated carrying capacities and then searches
for CE performance levels that allow the resulting environmental burdens to
comply with those allocated carrying capacities (when possible). Prepare the
same prerequisite inputs and completed MFA/LCA results for the starting foreground,
supply the allocated carrying capacities, then call `run_optimization`. A separate
`run_asr` call is optional.
The optimizer calculates ASRs for its candidates using the same assessment
definitions.

For both monitoring and target setting, allocated carrying capacities can be
calculated following the
{doc}`optional pyaesa tutorial <tutorials/optional_workflows/pyaesa_allocated_carrying_capacity>`
or supplied by the user. The pyaesa tutorial requires completed 2019 LCA results via `run_lca`.

## 1. Select a workspace

Run `set_workspace(...)` once at the beginning of each Python session.

```python
from pathlib import Path

from pytoce import set_workspace

workspace = Path("pytoce_workspace")
report = set_workspace(workspace)
print(report.as_dict())
```

`set_workspace` creates three package directories: `data_raw`, `data_processed`,
and `results`.
Runnable scenarios belong in
`data_raw/foreground_scenarios/scenarios/`. Allocated carrying capacity (aCC)
results belong in `data_raw/allocated_cc/results/`. The article and Methodological
appendix PDFs, the citation guide and the
[per-unit LCI workbook](publications_and_resources.md#lci-workbook)
are under `data_raw/methodological_notes/`. The workbook is also downloadable
from the package documentation.

## 2. Download public inputs

Place these licensed files under `data_raw/ecoinvent/` before calling `download_data`:

```text
ecoinvent_3.12_cutoff_ecoSpold02.7z
ecoinvent 3.12_LCIA_implementation.7z
```

Then run:

```python
from pytoce import download_data

download_report = download_data()
```

On first registration, `download_data` checks the archive size and SHA-256
against the packaged source catalog. LCIA extraction uses those authenticated
bytes and publishes the workbook. Later workflows reuse these sources and the
material recipe from their completion manifests.

`download_data` also writes `data_processed/ecoinvent/material_recipe.json` and its
adjacent guide. This local product records monomer requirements,
chemical-recovery yields, manufacturing input/output bases, carbon fractions
and the exact source identities derived from 35 licensed EcoSpold datasets.
Manufacturing records distinguish resin required per kg of finished polymer
from finished output per kg of the native processing service. MFA quantities
and allocated LCI exchanges have different meanings; the recipe describes their
normalization bases.

The packaged `ecoinvent_material_recipe_schema_v1.json` defines the product fields.

## 3. Prepare backgrounds

A premise decryption key is required to prepare new backgrounds from the
encrypted REMIND IAM scenario files. Request the key from the premise developers
as described in the [premise documentation](https://premise.readthedocs.io/en/latest/getting_started/preparation.html).
Pass it through `premise_key`:

```python
from getpass import getpass

from pytoce import prepare_backgrounds

background_report = prepare_backgrounds(
    premise_key=getpass("premise decryption key: "),
)
```

The default call prepares the four REMIND SSP2 background scenarios available in premise 2.4.6. They cover climate policy pathways ranging from below 1.5 °C to above 3 °C of warming by 2100. The warming range follows the indicative temperature classifications in the premise scenario catalog, as described in the article's Methodological appendix.

Pass one or more `background_scenario` names to prepare a subset. Later calls
reuse completed backgrounds.

The workflow verifies encrypted REMIND IAM files when it downloads them.
Subsequent background preparation uses those stored files directly.

General background data use ecoinvent through 2025, interpolate between
ecoinvent 2025 and premise 2030 for 2026--2029, and use the selected premise
scenario from 2030 onward.

Each of the 12 world regions identifies where the MFA records a material flow.
Its LCI may use
a dataset with different geographical coverage, such as RER, Switzerland or
RoW, when that dataset is the only available proxy for the modeled process. The
provider catalog declares each direct supplier's location. Provider records retain the selected dataset's actual location alongside its material and process meaning.

Background calculations use deterministic central inventories.

## 4. Run the MFA

```python
from pytoce import run_mfa

foreground = "frozen_2019_default_baseline"
mfa_report = run_mfa(foreground_scenario=foreground)
```

This example calculates 2019frozen_SSP2demand through its foreground scenario identifier
`frozen_2019_default_baseline`. Calculate `remind_mfa_default`
(publication name REMIND-MFA_default_SSP2) separately to compare the two foregrounds. Either foreground, or a user-defined
foreground with completed MFA and required LCA results, can be an optimization
parent.
The argument accepts one name or a sequence. History through
2019 is fixed, scenario changes begin in 2020, and `refresh=True` recalculates
only the requested MFA results.

User-defined foreground scenarios can change in-use stock targets, mean product
lifetimes, collection, treatment composition, and virgin feedstock composition.
The guide at `data_raw/foreground_scenarios/templates/README_foreground_scenarios.md`
in the workspace explains the accepted inputs and their limitations.

The MFA retains existing cohorts and calculates their outflows from the lifetime
distributions. The requested in-use stock must therefore be at least as large
as the stock surviving from earlier cohorts.

### Read the MFA tables

The report gives the file paths under
`results/<foreground_scenario>/mfa_results/`.
`resolved_annual_input_parameters.parquet` contains the annual foreground inputs;
`derived_annual_outputs.parquet` contains calculated stocks, flows, trade and
treatment quantities. `material_realization.parquet` reports the amounts sent to treatment, recovered
resin and monomers, virgin monomer use, and new polymer production, in Mt.
Select the quantity of interest using `flow_name`. Adding all rows would count
material more than once: for example, both the ethylene used in polymerization
and the resulting polyethylene are recorded.

```python
import pandas as pd

mfa_item = mfa_report.results[0]
annual_flows = pd.read_parquet(mfa_item.derived_annual_outputs)
```

## 5. Run the LCA

The LCA links production, conversion and treatment flows to
life cycle inventories (LCIs) and characterizes their environmental impacts.

```python
from pytoce import run_lca

lca_report = run_lca(foreground_scenario=foreground)
```

This call creates the LCA results used for monitoring or as the required baseline
for optimization. The
default selection uses all four background scenarios, EF v3.1 and PB-LCIA, and
upstream analysis years 2019, 2050, and 2100. Within one call, all selected
methods characterize the same inventory. You can also select an exact
registered Brightway method tuple. Calculations use central characterization
factors. Stored uncertainty parameters contribute to the method identity.

Upstream analysis years select detailed contribution reporting; annual and
period-total stage scores still cover 2019 through 2100. Optimization can reuse
a completed baseline with any upstream-year selection when its foreground,
background, and LCIA methods match. Conflicting scientific parents stop the
request; resolve the preserved result directories identified in the error.

Result directory names retain the selected background, method and analysis
years. Exact method labels preserve case and punctuation through compact
escapes. The assembled directory name may contain at most 212 characters,
leaving room for the temporary name used during refresh. If many separated
analysis years exceed this limit, split them across calls.

### Read the LCA tables

For each foreground/background selection, the report gives the paths under
`results/<foreground_scenario>/lca_results/<lca_selection>/`.

| File | Contents |
| --- | --- |
| `annual_scores.parquet` | Environmental burdens for each year from 2019 to 2100. |
| `scores.parquet` | Environmental burdens summed over 2019--2100. |
| `annual_life_cycle_stage_scores.parquet` | Annual environmental burdens separated by life cycle stage, such as production, conversion, recycling, or disposal. |
| `life_cycle_stage_scores.parquet` | Environmental burdens summed over 2019--2100 for each life cycle stage. |
| `contributions.parquet` | Contributions to environmental burdens for the years selected with `upstream_analysis_year`, with detail by supplying activity, elementary flow, or classification. |
| `stage_contributions.parquet` | Contribution detail for the same selected years, separated by life cycle stage. |

The `unit` column records the unit for each impact category. Global scores sum
the 12 regional scores for the same LCIA method, impact category, and year or
period. Calculate global shares from these global totals.

Select one `contribution_view` before adding contribution rows: the views are
alternative decompositions of the same burden. Native-activity and
elementary-flow detail are ranked separately by `contribution_norm`.
Each region/year/method group retains up to 100 entries per view; stage detail
retains up to 25 entries per view and stage. Each truncated view has its own
`exact_residual`. Separately labeled `modelled_direct` rows supplement the
native-activity entries, with their own contribution totals. Global detail is
selected from complete regional contributions before regional detail selection.

```python
lca_item = lca_report.results[0]
annual_impacts = pd.read_parquet(lca_item.annual_scores)
```

In `contributions.parquet`, `share` divides a signed contribution by the LCIA
total for the same region, year and method. In `stage_contributions.parquet`,
it uses the corresponding stage total. `contribution_norm` sums absolute
characterized contributions in the category's unit. `norm_share` uses the corresponding whole-result or
stage-specific absolute total. Positive and negative contributions can cancel,
making `share` unavailable while `norm_share` remains defined if its denominator
is nonzero. Read the accompanying status:

| Status | Meaning |
| --- | --- |
| `available` | The denominator can be used. |
| `zero_total_activity` | Provider demand is zero. |
| `near_zero_total_activity` | Signed provider demands nearly cancel: `effective_unit_score` is unavailable, while `effective_unit_norm` remains defined when absolute demand is nonzero. |
| `zero_contribution_norm` | The sum of absolute characterized contributions is zero. |
| `zero_score` | The LCIA total is zero. |
| `near_zero_score` | Positive and negative impacts nearly cancel. |
| `not_applicable` | Provider-level normalization is inapplicable to this row. |

The README beside these LCA tables gives the
remaining field definitions and units.

## 6. Assess absolute environmental sustainability for monitoring

Calculate allocated carrying capacities through the pyaesa tutorial workflow,
or supply your own values based on your chosen allocation method and shares.
Place a complete aCC result under `data_raw/allocated_cc/results/<name>`.
Its YAML identifies the annual and cumulative CSV files, following the installed
templates. Each row records its source in `cc_authority`. Inputs distinguish
annual steady-state values from cumulative dynamic climate change values.
The optional pyaesa tutorial propagates selected sources of uncertainty through
Monte Carlo sampling to obtain allocated carrying capacity distributions.
P5, P50, and P95 are the 5th, 50th (median), and 95th percentiles of these
distributions. Each percentile is used as a separate
deterministic allocated carrying capacity value in the ASR calculations. You can
also supply results from a deterministic allocation or other distribution
statistics, such as the mean. Record the allocation method, statistic, and data
source for each input.

Then compare the completed environmental burdens with their allocated carrying
capacities:

```python
from pytoce import run_asr

asr_report = run_asr(
    foreground_scenario=foreground,
    allocated_cc_result="name_of_your_allocated_cc_result",
)
```

Use the same `lcia_method` selection as the completed LCA: `"ef31"` or both
`"ef31"` and `"pb_lcia"` (the default). The assessment uses the sixteen EF v3.1
categories. Select backgrounds and aCC cases or retain the defaults
of all four backgrounds and all available cases. Each foreground/background
pair has its own result directory under `results/<foreground>/asr_results/`.

`annual_asr.parquet` divides environmental burdens by the corresponding annual
aCC for each region, category, year, and case. An ASR <= 1 indicates compliance
with the allocated carrying capacities, whereas an ASR > 1 indicates
transgression. Annual results include 2019 for reporting;
`assessed_for_compliance` marks the 2020--2100 assessment years.

`cumulative_gwp100_asr.parquet` additionally assesses climate change over
2019--2100. Its numerator sums annual all-GHG GWP100 burdens, and its denominator
is the independently supplied cumulative aCC.

The {doc}`monitoring tutorial <tutorials/core_workflow/5m_run_asr>` shows how to
read both assessments. Repeated calls reuse completed results; `refresh=True`
recalculates only the selected ASR results. Completed aCC inputs are immutable:
supply corrected allocated carrying capacities under a new `allocated_cc_result` name.

## 7. Run the optimization

Optimization uses the same complete allocated carrying capacity input contract
as monitoring. The allocated carrying capacities can be calculated through the
pyaesa tutorial workflow or supplied by the user.

Completed MFA and LCA parents are reused by name when their manifests are
present. To apply changed scenario inputs, refresh the affected
workflow products. After a scientific method change that affects MFA or LCA,
calculate the affected parents in a new workspace. The same parents can be reused across optimization grids and refinement sequences.

```python
from pytoce import run_optimization

optimization_report = run_optimization(
    foreground_scenario="frozen_2019_default_baseline",
    allocated_cc_result="name_of_your_allocated_cc_result",
    candidate_workers=1,
)
```

Replace `name_of_your_allocated_cc_result` with the directory name of a
complete result under `data_raw/allocated_cc/results/`.
`foreground_scenario` accepts one name or a sequence of names. Each requested
foreground is optimized independently and its results remain in its own
`results/<foreground_scenario>/optimization_results/` directory.

`optimization_lcia_method="ef31"` uses all 16 EF v3.1 impact categories and
cumulative GWP100 for the constraints, search, ranking, and pathway selection.

`lcia_method` chooses the impact categories used to report environmental burdens
for the selected CE transition pathway. It accepts `"ef31"`, `"pb_lcia"`, both
groups, or the full Brightway method tuple for one of the package's 16 EF v3.1
or ten PB-LCIA impact categories. Supplying one full method tuple reports
burdens for that category in the selected pathway's environmental burden tables.

`allocated_cc_case` accepts one or more case names; `None` selects every available
case independently. Case names identify the supplied inputs; their definitions
and sources describe the allocation methods and statistics used.

`background_scenario=None` selects all four supported REMIND SSP2 backgrounds.
Pass one readable name or a sequence, such as
`("remind_ssp2_npi", "remind_ssp2_pkbudg650")`, to score candidates and select
and report a pathway for each requested background.

The optimizer requires complete LCA results for the foreground being optimized
under each selected background. Run `run_lca` first with the same `lcia_method`
selection as `run_optimization`: EF-only, PB-only, both groups, or one supported
impact category. These results supply the baseline for reporting the selected
pathway. The prepared EF methods supply the optimization constraints.

The shared response bank covers all four supported backgrounds. Building it
requires all four prepared backgrounds, including for requests selecting a
subset. Once the bank is complete, the optimizer loads and scores the requested
background responses. Complete any required background preparation with
`prepare_backgrounds` before running optimization.

`candidate_workers` accepts integers from 1 through 61 and defaults to one.
Choose a count that fits available CPU and memory for the workers and coordinator.
Changing the count configures execution for the next call, which can reuse
matching completed candidate calculations. Progress counts completed evaluations
in the current batch, including candidates that violate MFA constraints.

### Search blocks and refinement

The search varies CE performance target levels and constructs an annual MFA trajectory
for each tested combination before estimating environmental burdens.
Demand reduction scales the parent's in-use stock demand in each year. Its
value expresses the change in the modeled stock target. Lifetime
extension changes the mean and standard deviation proportionally. The same
relative demand and lifetime changes apply across all regions and end-use groups.
Collection, treatment composition, and virgin feedstock composition reach common
absolute target shares while starting from each region's parent values.
Monotonic demand reduction and lifetime extension are study assumptions.

The search first checks the unchanged parent and retains it where it already
complies. Otherwise, the three blocks explore CE performance target levels:

| Block | CE performance families varied | Default initial exploration |
| --- | --- | --- |
| I | Demand reduction and lifetime extension. | All 36 combinations of demand reductions of 0/20/40/60/80/90 percent and lifetime extensions of 0/20/40/60/80/100 percent. |
| II | Collection, treatment composition, and virgin feedstock composition. | A pairwise covering design of 132 combinations, plus the parent. |
| III | All five families together. | A separate pairwise covering design, combinations derived from Blocks I and II, and intermediate combinations tested through path relinking. |

A pairwise covering design represents compatible pairs of parameter levels.
Each resulting combination is evaluated through the MFA.
Refinement tests finer changes around retained combinations, using the updated
Block I and II results before refining Block III. Treatment and virgin feedstock
shares each continue to sum to 100 percent. The search evaluates the proposed points along each tested direction, including
points beyond an infeasible or worse-performing intermediate combination.

The search retains the best ranked combination among those evaluated. Its results describe the combinations evaluated by the finite search. The
[optimization architecture note](https://github.com/Ike-EDB/pyToCE/blob/main/src/pytoce/run_optimization/ARCHITECTURE_run_optimization.md)
describes the starting designs, path relinking, refinement, and stored search
records for contributors.

`initial_grid_step_pct=20` selects the initial coarse spacing; supported values
are 10, 20, 25, 50 and 100 percentage points. Smaller initial spacing increases
the starting design.
`endpoint_refinement_steps_pct=(10, 5)` selects a nonempty, strictly decreasing
sequence of positive integer steps smaller than the initial spacing; `(5,)`
requests only five-point refinement. The default is therefore 20, then 10,
then 5. These steps control endpoint proposals and refinement. Annual demand repair uses
a separate one-percentage-point grid.

To add finer refinement after a completed run, keep the same scientific inputs
and reporting methods, including the same `allocated_cc_result`,
`background_scenario` and `lcia_method` selections. Pass the complete extended
sequence, for example:

```python
finer_report = run_optimization(
    foreground_scenario="frozen_2019_default_baseline",
    allocated_cc_result="name_of_your_allocated_cc_result",
    optimization_lcia_method="ef31",
    lcia_method=("ef31", "pb_lcia"),
    initial_grid_step_pct=20,
    endpoint_refinement_steps_pct=(10, 5, 1),
)
```

A compatible completed sequence such as `(10, 5)` supplies its completed stages;
the extended request adds the new stage and reuses matching candidate calculations.
Keep the same initial spacing, scientific inputs, reporting methods, and executed
blocks. Reuse also requires compatible search, ranking, and pruning methods;
matching step sizes alone is insufficient. New candidates, completion-in-2100
evaluations, and final reporting can still require calculations. Each completed
sequence retains its own result directory.

### MFA feasibility and demand-timing repair

For each tested combination, the fraction of change from the parent toward the
selected CE performance target levels increases linearly from zero in 2019 to
one in 2050 and remains one through 2100. Parameters outside the active block
retain their parent trajectories.
The complete MFA checks stocks, trade, recycling, and the use of recovered
materials in every year. A linear trajectory that satisfies the MFA constraints is retained
even when its environmental burdens exceed the allocated carrying capacities.

If the trajectory is infeasible in the MFA and demand reduction is active,
demand-timing repair may change only the timing of demand reduction before 2050.
The target level, completion year, lifetime trajectory, collection, treatment
composition, and virgin feedstock composition remain fixed. A linear program
minimizes the mean absolute difference from the linear demand-reduction
trajectory over 2020--2049 while preserving monotonicity and stock constraints
through 2100. Its solution must then satisfy all stock and material-flow constraints in the
complete MFA. If needed, a finite set of additional annual demand trajectories
is tested in order of increasing departure from the linear reference.

Trajectories that satisfy the MFA constraints proceed to the environmental
assessment. Repair retains the existing cohorts and the modeled treatment
routes. It selects a demand trajectory using the stock constraints and full MFA
validation. An unresolved repair excludes the tested combination after the
available alternatives have been evaluated. The linear-program solution applies
to its conditional stock problem, and the additional repair search covers its
declared alternatives.

Intervention distance compares CE performance target levels with the requested
parent. Collection, treatment, and virgin feedstock shares use the parent's
values in the completion year; demand reduction and lifetime extension are
fractions of the parent values. The CE performance families active in each block
have equal weight. Two trajectories can have equal intervention distance and
different annual values. The distance reports changes in the selected endpoint values; annual
trajectories and MFA feasibility are reported separately.

`declared_neighborhood_verified` reports whether the declared final set of
nearby changes was fully resolved, with the selected candidate retained as best
and all demand-repair outcomes resolved. MFA feasibility, environmental compliance, and completion of the declared
neighborhood checks are reported separately. The architecture note
defines the tested changes for each block.

For later completion in 2100, recalculate the transition fraction to reach one
in 2100 for the same selected target levels and apply the same demand-only
repair. Controls outside the active block retain the parent; active circularity
transitions use each calendar year's parent values and the revised completion
weight. An unchanged parent is reused. These timing evaluations retain the
selected endpoints. MFA feasibility checks remain annual through 2100, with
adjustable demand years 2020 through the year before completion.

### Selection rules

Candidate selection is performed separately for each search block, prospective
background system, and allocated carrying capacity input. The combinations
considered satisfy the MFA constraints. Absolute environmental sustainability
compliance requires ASR ≤ 1 for each region and each studied year from 2020 to
2100 for the steady-state EF 3.1 carrying capacities, and over the cumulative
2019–2100 period for the dynamic climate change carrying capacity.

The unchanged parent is retained when it is compliant. Otherwise, the criteria
below are applied in the order shown. The next criterion is used only if two or
more combinations have the same result for the preceding criteria.

**Compliant selection**

| Order | Selection criterion |
| --- | --- |
| 1 | Minimize the CE intervention distance for the corresponding block. |
| 2 | Retain the combination whose highest ASR is closest to one. |

**Noncompliant fallback**

| Order | Selection criterion |
| --- | --- |
| 1 | Maximize the number of fully compliant regions. |
| 2 | Maximize the number of fully compliant impact categories. |
| 3 | Maximize the number of compliant region–impact category pairs. |
| 4 | Maximize the number of individual compliant ASRs. |
| 5 | Minimize the maximum positive ASR exceedance. |
| 6 | Minimize the mean positive ASR exceedance. |
| 7 | Minimize the CE intervention distance for the corresponding block. |

For the noncompliant fallback, a region is fully compliant only if all 16 annual
impact categories and cumulative GWP100 comply over their respective assessment
periods. An impact category is counted as fully compliant across regions only if
the corresponding assessment complies in every region, with cumulative GWP100
counted separately from annual climate change. The region–impact category
criterion applies these 17 assessments separately to each region, giving 204
region–impact category pairs. The individual compliant ASR criterion counts
every annual region–impact category–year comparison and one cumulative GWP100
comparison per region, giving 15,564 comparisons.

Positive ASR exceedance is `max(ASR - 1, 0)`. The positive ASR exceedances for each
annual impact category are first averaged over the 81 assessed years. The
resulting 16 annual values and the one cumulative GWP100 value receive equal
weight within each region, followed by equal weight across the 12 regions.

### Read the optimization tables

Use the selected-product paths returned by `optimization_report.products`.
The guide at `results/<foreground_scenario>/optimization_results/README.md`
describes the table-specific identity joins, periods, statuses and units.
Use those join keys: stage-table `candidate_identity` records the MFA candidate
identity, whereas coordinate and annual-control tables use the
selection candidate identity.

`selected_coordinates.parquet` reports endpoints and MFA validation;
`selected_annual_interventions.parquet` contains the annual controls as
fractions. Join these tables by `candidate_identity` and read each path's
`physical_status`. Attempted paths remain recorded for unsuccessful MFA outcomes;
environmental burdens and absolute sustainability ratios (ASRs) are reported for
paths that satisfy the MFA constraints.

The `selection_status` column describes environmental compliance:

| Status | Meaning |
| --- | --- |
| `baseline_compliant` | The unchanged parent meets every aCC limit. |
| `intervention_compliant` | The selected intervention meets every aCC limit. |
| `selected_noncompliant` | The best tested alternative under the ordered criteria still exceeds at least one limit. |

Read each foreground, background and allocated carrying capacity input separately.
Selection status reports the choice among evaluated candidates, MFA validation
reports material-flow feasibility, and local neighborhood verification reports
completion of the declared nearby tests. Use the saved annual controls to read
a repaired demand path. The adjacent guide explains repair statuses and the units
and process boundaries of the feedstock tables. Keep all selection-role links
and count each distinct MFA result once when aggregating joined tables.

## Reuse and refresh

A completed manifest marks a result as ready for reuse. After changing an input
under the same name, use the workflow's `refresh=True` option where supported,
or choose a new name. If a completed
result is damaged or inconsistent, keep it unchanged and calculate a result
under a new name.
