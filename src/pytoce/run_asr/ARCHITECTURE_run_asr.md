# Monitoring ASR architecture

`run_asr.py` owns the public request and result reports. It imports scientific
dependencies only when the function is called. `results.py` reads completed LCA
results and supplied allocated carrying capacities, admits matching methods,
factors, units, and coordinates, and writes the monitoring tables and manifest.

Monitoring and target setting call `shared.assessment.asr.calculate_asr`.
This owner calculates annual ratios, cumulative GWP100 burdens and ratios,
and global metrics. Global numerators and denominators use separate stable sums.
Both workflows use the same arithmetic and summation order.
Candidate ranking, intervention distance, and selection remain in optimization.

The existing `run_optimization/input/allocated_cc*` modules remain the canonical
owners of aCC validation, compilation, case selection, and immutable
processed inputs. Monitoring uses those input owners directly. `run_lca/calculation/result_inputs.py`
selects completed LCA parents for both monitoring and optimization independently
of optional upstream contribution years, rejecting conflicting scientific parents.

Each monitoring result covers one foreground/background pair and selected aCC
cases. Annual tables include 2019--2100 and assess compliance over 2020--2100
using the 16 EF v3.1 categories. Cumulative GWP100 tables cover 2019--2100
with a separately supplied denominator.

Results are written under `results/<foreground>/asr_results/<selection>`. A writer
lock protects the result scope. Publication stages files beside the destination,
writes the completion manifest last, and replaces only completed result scopes
containing ordinary package files. Completed results are reused from their manifests. Refresh
recalculates the selected ASR results.

Focused tests live in `tests/unit/run_asr/`. They cover annual and cumulative
definitions, signed burdens, global aggregation, method compatibility, input
damage, reuse, atomic refresh, and publication failure. Optimization metric and
parent-selection tests cover the direct consumers of the extracted owners.
