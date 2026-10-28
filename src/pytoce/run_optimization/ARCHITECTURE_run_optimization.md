# `run_optimization` architecture

`run_optimization.py` validates the selected foreground and background scenarios,
allocated carrying capacity input, impact methods, optimization blocks, cases,
initial grid spacing and refinement sequence. It uses completed MFA and LCA
results and returns `RunOptimizationReport`.

## Owners

- `input/allocated_cc_paths.py` resolves input paths and validates YAML metadata.
- `input/allocated_cc_contract.py` validates the annual and cumulative tables.
- `input/allocated_cc_product.py` compiles the aCC input and its identity.
- `calculation/interventions.py` converts candidate percentages to annual MFA
  assignments and retains their complete regional circularity values.
- `calculation/candidate_authority.py` constructs interventions from trusted
  baseline inputs and supplies their identities.
- `calculation/candidate_projection.py` applies dated responses and calculates
  candidate scores.
- `calculation/pa_candidate_projection.py` characterizes PA events
  through the shared native receiving inventory and selected response methods.
- `shared/assessment/asr.py` owns the annual and cumulative ASR calculation used
  by both monitoring and target setting, including stable global aggregation.
- `calculation/regional_metrics.py` uses that assessment to calculate compliance
  counts, exceedances, binding constraints, and ranking metrics.
- `calculation/demand_lifetime_search.py` owns the conditional closest-to-linear
  demand projection and one finite demand-only repair pool.
- `calculation/candidate_search.py` owns endpoint refinement for Blocks I and II
  and the ranking shared by all blocks. `calculation/integrated_design.py` and
  `calculation/integrated_search.py` own the joint design and refinement for III.
  `calculation/request_search.py` completes I/II before III at each spacing and
  retains completed stages for continuation. Annual demand timing is determined by the MFA repair calculation.
- `calculation/candidate_worker.py` constructs one MFA representative per
  endpoint task, then scores the admitted state once. Its protocol,
  stored inputs, and stored outputs have separate modules in the same folder.
- `runtime/scheduler.py` limits concurrent candidate calculations to
  `candidate_workers` (default one; supported range 1--61), with one
  numerical-library thread per worker. The coordinator owns cache writes and
  returns summaries in task order. After all selected timing calculations finish,
  workers close before full selected results are loaded for reporting.
- `runtime/candidate_results.py` saves completed candidate results so a repeated
  request reuses completed candidate calculations.
- `calculation/response_builder.py` creates response banks from the prepared
  backgrounds, provider services and LCIA methods.
- `calculation/selected_reporting.py` calculates changes to annual and cumulative
  stage results. `calculation/selected_stage_reporting.py` streams those results
  one MFA candidate and background at a time.
- `calculation/results.py` assembles the selected product, including actual
  annual intervention paths, requested/realized identities and repair outcomes.
- `runtime/result_store.py` writes result tables and the final manifest.

The explicit fields of the selected tables account for most of `results.py`.
Keeping their assembly together makes the joins between selections, MFA outcomes,
annual controls and table identities visible in one place. MFA calculation,
characterization, stage streaming and file writing have separate owners.

## Execution and Progress

`candidate_workers` is a positive integer from 1 through 61, defaulting to one
worker. The upper bound matches the Windows process-pool limit. Set it to three,
for example, only when available CPU and memory can accommodate three
workers and the coordinator. Each worker retains its own scientific inputs and
MFA state. Submitted work and completed-result buffering are bounded, and
the worker pool closes before full selected results are assembled.

The worker count controls concurrency. Changing it can reuse completed matching
calculations. Processing-time totals record per-worker durations and can overlap
when workers run concurrently.

During each candidate batch, the progress display counts completed evaluations
and distinguishes newly calculated results from reused results. It updates as
completed results become available, replacing the displayed line in a terminal
or notebook and writing separate lines when output is redirected. A completed evaluation reports the candidate's MFA outcome and, for admitted
candidates, its environmental assessment. The denominator and percentage describe
the current batch. Later batches depend on earlier search results.

## Optimization method

The workflow uses allocated carrying capacity values calculated through the
pyaesa tutorial workflow or supplied by the user. Case names identify separate inputs. Their source definitions specify the
allocation methods and statistics used.

Stock-target reduction ranges from 0 to 90 percent; mean-lifetime extension
ranges from 0 to 100 percent. Environmental search operates on endpoint
coordinates. Each task uses a transition fraction that increases linearly from
zero in 2019 to one in the target year and remains one afterward. Demand and
lifetime changes scale the corresponding parent values in each year; shares
move from each year's regional parent values toward the common absolute target.
Inactive controls retain the parent.

### Initial Exploration and Refinement

The exact unchanged parent is evaluated first. If it is compliant for every
selected background and allocated carrying capacity input, that block retains it as the block's completed selection. Otherwise, with the
default initial grid, Block I starts with all 36 demand/lifetime endpoint
combinations: demand levels
0/20/40/60/80/90 percent and lifetime-extension levels 0/20/40/60/80/100 percent.
Block II uses 132 circularity/feedstock settings plus the parent, selected from the
20-point coarse grid to cover all 873 composition-admissible pairs of levels.
Block III compares both single-block alternatives and joint interventions under
its own common ranking, which gives equal weight to all five intervention
groups. Block III compares interventions using its common five-family distance.
It retains the actual I-only and II-only selections, their direct combinations,
and every coarse demand/lifetime pair combined with each distinct II circularity
selection. Each combination receives its own MFA calculation and admission.

Independent joint starts cover every algebraically compatible pair of coarse
levels among demand, lifetime, collection, the four treatment shares and the
three feedstock shares. Bounds and both exact share totals constrain this
coverage. The design is generated progressively from compatible pairs. It covers pairs of endpoint levels. Each resulting combination receives its own
MFA feasibility check. Parent circularity remains a separate category with
its actual regional and annual values. A separate demand/lifetime starting
domain retains that category, while explicit circularity starts remain eligible
even when a parent-circularity candidate currently ranks best.

At the coarse spacing and each refinement stage, Blocks I and II provide their
current selections before Block III develops joint alternatives. Distinct earlier
I/II selections remain anchors. The independent design also includes the zero
demand/lifetime version of each of its circularity vectors; actual I choices are
crossed with every circularity vector represented by that design or an actual II
selection. The base is admitted before choosing guides. For each state without
a compliant unchanged parent, guides are its best admitted explicit independent candidate and its best
admitted explicit base candidate at each actual I demand/lifetime pair, together
with actual explicit II anchors. Each guide comes from an admitted candidate. Guides remain fixed throughout the stage. Parent circularity is retained as a categorical choice.

The routes change circularity alone; demand, lifetime or both with circularity
held fixed; or each of those three demand/lifetime choices together with
circularity. These seven projections can strengthen or weaken interventions.
Each scalar moves toward its target by at most the initial spacing. Each
treatment and feedstock partition transfers at most that many percentage points
from surplus to deficient shares, in fixed field order. Both partitions continue
to total 100, including bounded final moves to off-grid targets. Each projected
segment is traversed in both exact orientations. Separate circularity-only
segments reach collection bounds and each donor-to-recipient share boundary at
fixed demand/lifetime. Dedicated segments also connect each II choice to the
actual I demand, lifetime and combined values while retaining that II circularity.
The initial spacing remains fixed for these complete routes: finer stages add routes from new anchors and guides. Every
distinct point through the exact route target is proposed, including after a
MFA rejection, failed demand repair or a worse score. Numerical execution
errors still stop the request. Only admitted points receive scores.
These routes change the requested endpoints. Each endpoint uses the common demand-repair method.

For each state, Block III locally refines the best admitted candidate in each of
two domains: parent circularity and explicit circularity. Local refinement starts from an admitted candidate. The coarse winners await the first refinement; each subsequent
stage selects its domain winners from all accumulated observations, including
new starts and routes. Scientific proposal-family minima remain recorded to
describe where good candidates were found, as provenance. All scored candidates remain eligible for overall selection.


Blocks II and III generate complete finite scalar and donor-to-recipient transfer
lines at each refinement spacing. Scalar lines contain all admissible positive
and negative step multiples from the incumbent, together with both exact bounds.
Each share-transfer line contains all admissible step multiples and the exact
point where the donor share reaches zero; the partition still totals 100.
Proposals are evaluated subject to the distance-pruning policy below; unpruned
terminal polls evaluate the complete lines. Every evaluated proposal is subject
to the unchanged MFA feasibility rules, with proven equivalent admission or
repair work reusable across requests. The full proposed line remains available after a worse score, MFA rejection, or failed demand repair. Block I
retains its adjacent demand/lifetime polling.

Line polling is useful because the ordered compliance counts can worsen and then
improve along a control direction, even when some individual burdens vary
monotonically. Material receiving constraints and the demand-repair construction
also prevent a general monotonicity assumption. Finite line enumeration evaluates the sampled points separately, including
points on either side of a rejected or lower-ranked point.

Block III processes fixed batches in control-group order: demand/lifetime lines,
collection, treatment transfers and feedstock transfers. At a scalar stall,
additional one-step demand/lifetime diagonals, demand/lifetime changes combined
with each circularity group, and moves toward the frozen guides are tested.
These additional moves use the declared one-step combinations.
The best strict improvement in the first successful batch restarts scalar polling;
later batches can be skipped. A terminal search completes the whole declared
neighborhood. The batch order is fixed.
Refinement continues until the current winner in every state and circularity
domain has matching terminal evidence, including after another state's search
changes a previously checked winner. Each stage retains the current state/domain winners. All accumulated observations remain available at the next stage.

The method combines [categorical-variable treatment](https://optimization-online.org/2001/03/303/),
finite [path-relinking constructions](https://www.uv.es/~rmarti/paper/docs/ss1.pdf)
and direct search over discrete controls. Work on
[derivative-free integer optimization](https://www.iasi.cnr.it/~liuzzi/papers/2020_MPC_Liuzzi-etal.pdf)
motivates looking beyond immediately improving steps. The package's complete
finite lines, ordered compliance objective and MFA feasibility rules are
specific adaptations; they define the finite proposal set evaluated by this implementation. Selection compares the candidates evaluated by these starts, routes, and refinement steps. A terminal result records stability over its declared tested proposals. The demand-repair grid remains one percentage point. See
[the optimization tables](../../../docs/workflow_reference.md#read-the-optimization-tables) for the saved coordinates, annual fractions
and completed-stage evidence.

### Refinement Pruning

In Blocks I and II,
the final requested step and every requested step of five points or finer
evaluate their declared neighbors;
appending a one-point step retains this behavior at five points. Block III
retains each state/domain winner's declared polling obligations at every spacing.

Rational arithmetic preserves exact annual control values; assignments
and reported values remain floating point.

### MFA feasibility and demand-timing repair

A complete MFA admits the linear reference. If feasible, retain it
unchanged even when environmentally noncompliant. Only demand may be repaired;
lifetime and active circularity/feedstock transitions remain fixed. Zero demand
reduction has a unique monotone path, and Block II retains the parent's demand trajectory.
Lifetime dispersion remains proportional to the mean.

After an MFA rejection with nonzero demand, the initialized MFA session
supplies fixed-lifetime cohort coefficients to one stock LP. It minimizes mean
absolute demand departure from linear over 2020 through the year before
completion. Annual stock constraints extend through 2100. A solver-reported
infeasible result concerns that fixed-control formulation. Solver/model errors
remain execution errors. The LP's variables describe control increments and
absolute departures from the linear reference.
The ordinary and candidate MFA use the same cohort roundoff convention.

The one LP proposal must pass complete MFA admission. If rejected, retain its
path and diagnostic and construct one immutable demand-only pool around it.
Adjacent annual levels combine the absolute one-percentage-point grid, exact
linear-reference values and exact anchor values. Scalar moves cover
every adjustable year; paired moves connect the diagnosed binding year, capped
at the last adjustable year, with each earlier adjustable year. Monotone
propagation preserves endpoint bounds. Deduplicate and sort by mean absolute
demand departure, then canonical annual values; retain the first fully feasible
path. The pool remains fixed during evaluation. Pool exhaustion is recorded as `repair_unresolved` after the declared pool has
been tested.

Repair performs MFA calculations only. The admitted MFA state is retained
for one subsequent activation/environmental evaluation, shared across the
request's backgrounds and aCC cases. An admitted LP projection is a numerical
minimum-departure solution of its conditional stock problem. Tied LP optima may
remain unexplored. Local-pool success establishes nearestness only in that pool.
Inner attempts retain the diagnostics used by the repair calculation.

After selection at 2050, recalculate the transition fraction to reach one in
2100 for the same target levels and apply the same demand-only repair. These timing calculations retain the selected target levels. All stock and
material-flow constraints remain annual through 2100.

A delayed path can be infeasible in the MFA even when its 2050 path is feasible.
The selected binding preserves that saved outcome, including its diagnostic code.
`calculation/results.py` partitions outcomes once: every requested path retains
coordinates and selection roles, while only feasible candidates enter
aCC, PA, environmental, stage and feedstock reporting. The 2050 selection remains fixed. Repair outcomes distinguish linear, stock-projection
and local-pool feasibility from fixed-control stock infeasibility, unique-path
infeasibility and failed demand repair.

With `n` adjustable years, one repair pool has at most `2*n + 4*(n - 1)` paths
before deduplication and admission. Including the linear reference and stock
projection, this bounds full-MFA calls by 178 and 478 for completion in 2050 and
2100 respectively. A feasible linear path needs only one check; reuse can avoid
it. There is at most one stock LP call and one environmental evaluation per
uncached endpoint setting. These bounds apply to each endpoint setting.

Treatment and feedstock shares are nonnegative and each group sums to 100 percent.
A candidate complies when every controlled absolute sustainability ratio is at
most 1.0. Selection retains the exact intervention-distance, compliance-coverage,
exceedance and identity ordering. Scalar stalls trigger coordinated endpoint
alternatives for Blocks I and III, with guided moves for explicit III
circularity. The `declared_neighborhood_verified` field requires the declared neighborhood
around the same final selection to be complete with every required demand repair successful. For III with
parent circularity, this is the declared demand/lifetime neighborhood; separate
explicit-circularity starts and branches remain recorded. A compliant unchanged
parent terminates with the parent shortcut recorded.

The final default endpoint step is five percentage points. Its neighborhoods
remain unpruned when a later one-point step is appended. A changed winner needs
a new final check. Stability, proposal resolution, and declared-neighborhood coverage are recorded
separately. Selected
coordinates and annual intervention fractions make the actual paths explicit.
The annual intervention table has candidate, year, region and parameter axes.
Demand/lifetime fractions have a null region and cover 2019--2100. The eight circularity fractions retain all 12 world regions for 2020--2100;
2019 remains the unchanged baseline. The trajectory owner supplies both the six
editable MFA assignments and the complete eight-share table from the same arrays.
An exact-parent path references the admitted regional parent values directly.
An unsuccessful timing retains the returned diagnostic trajectory, which can be
the rejected stock projection. Read it with its MFA and demand-repair statuses. Realized quantities and
environmental scores are reported for admitted trajectories.

## Storage and reuse

### Baseline LCA Results

Upstream analysis years select detailed contribution reporting; annual and
period-total stage scores still cover 2019 through 2100. Optimization can reuse a
completed baseline with any upstream-year selection for the required foreground,
background and LCIA methods. If several reporting variants have the same
scientific parents, it prefers the default-year result, then the first readable
path in lexical order. Conflicting scientific parents are ambiguous and stop
before the search; the error identifies the preserved result directories.

### Candidate and Selected Results

Each request has one candidate cache in its readable MFA calculation
directory, currently `<readable_request_selection>__stock_lp_v6/`, beneath the
aCC result and optimization-method selection. Completed searches have separate
leaves such as
`selected_results/joint_lines_v1_stock_lp_v6__ef31__grid_20pct__refinement_10_5pct/`.
The complete refinement sequence is part of the selected-product identity.
Candidate reuse requires matching scientific, scoring and metric inputs; matching
endpoint values or directory labels alone is insufficient.

The candidate cache authority binds MFA inputs, prepared receiving parents,
and the EF search projection, including mandatory climate, independently of
reporting methods or the containing response bank. PA score arrays use the EF
search methods. Stored PA events allow selected
results to be characterized with requested reporting methods. Reporting PA parent and candidate scores are calculated once and
shared by annual and stage results; matching EF scores reuse the search payload.
Selected product identities retain the requested reporting methods and sources.

The single candidate cache binds evaluation inputs and methods independently of
the initial grid and refinement schedule. It stores one completed endpoint
result with its manifest, including unsuccessful outcomes. A transient map
resolves completed rows by requested identity. Requested task
identity binds the parent, endpoints, target year, fixed controls and repair
method. The returned MFA/score identity binds the actual admitted annual
controls. The coordinator maps the requested intervention to its returned
representative; use the identity corresponding to the requested task or realized path. The typed
repair result retains MFA/LP/pool counts, admitted demand departure and
linear/stock-projection diagnostics.
The candidate result retains the repair outcome.
If no completed candidate directory exists,
its calculation is performed again; the calculation writes a new staged product.
A visible candidate directory without its manifest is preserved and reported
as incomplete. Public numerical tables contain selected candidates. The method
summary also retains compact completed-search records, including proposed and
skipped candidates, evaluated outcomes, guides, branches and completed stages.
The final result manifest is written last. Refresh applies only to
the selected optimization result and retains its existing inputs.

Selected result leaves retain the initial spacing and entire refinement sequence.
A completed full-schedule result wins immediately. Otherwise, a compatible
completed strict-prefix result at the same initial spacing supplies its completed
stages and compact worker records. Reuse requires matching calculation inputs,
selection and joint-search methods, executed blocks, and earlier pruning
policies. The compact completed-stage records retain this state. The request
restores I/II search state, III branches and guides, and accumulated anchors,
then appends only the missing stages. Its parent product identity remains an
explicit search/result input; earlier selected products remain unchanged.
Compatible individual candidate calculations remain reusable across stage policies.

Stage reuse requires matching scientific inputs, reporting methods (`lcia_method`),
search and ranking methods, executed blocks, initial grid and stage pruning
policies. The stage pruning policy also matters: a standalone final
ten-point stage is unpruned,
whereas ten-point polling within a longer I/II schedule may use distance pruning.
Matching individual candidate calculations remain reusable. The default five-point completion and its
one-point extension share compatible earlier policies.

### Completed Search Records

Completed-stage records distinguish the starting design, algebraic pair coverage,
finite routes, frozen guides, domain polls and terminal incumbents. The
`family_seeds` field describes family minima and proposal origins; it records search provenance. Coarse `pending_branches` and refined
`terminal_branches` identify the searched winners for each state in the parent-
and explicit-circularity domains. Coarse pending winners await local refinement. Poll records distinguish evaluated from skipped proposals and record the
coverage reached before an opportunistic improvement.
`admitted_starting_candidate_identities` includes admitted route points even when
they receive no local refinement. A fully compliant parent can end the search
early, as recorded by `parent_shortcut`.
