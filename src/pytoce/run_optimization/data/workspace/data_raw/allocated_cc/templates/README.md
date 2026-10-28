# Allocated carrying capacity (aCC) input

This directory contains templates for carrying capacities already allocated to
the modeled plastics system.

## Prepare an aCC result

Calculate allocated carrying capacities through the pyaesa tutorial workflow,
or supply your own values based on your chosen allocation method and shares.
 Calculate the aCC by applying
the allocated share to its carrying capacity (CC):

```text
aCC = aSoCC * CC
```

pyToCE checks the input fields, coverage of the 12 world regions, years, impact
categories, units, and values before using it.

The annual table uses these fields:

| Field | Required content |
| --- | --- |
| `allocated_cc_case` | A unique lowercase `snake_case` name. The source documentation defines the allocation method and statistic represented by the name. |
| `lcia_method` | `ef31`, the LCIA method used for monitoring and target-setting ASRs. |
| `region` | One world region. Document any mapping to the 12 world regions with the source files. |
| `impact_category_id` | The matching pyToCE impact category. |
| `year` | One year from 2019 through 2100. |
| `allocated_cc` | A positive aCC value. |
| `unit` | The native unit of the matching LCIA category. |
| `cc_authority` | A citation or source record for the carrying capacity value and its regional mapping. |

Each annual row and cumulative row records its source in `cc_authority`. The
compiled result also records the source file hashes. The YAML file identifies the result and its table files.

pyaesa is one possible source of an aCC result. When pyaesa is used, stage its
final aCC values and cite the exact pyaesa version, UNCASExt article,
Appendices B and C, and input sources required by that result. For other sources, record the corresponding method and data provenance.

Create
`data_raw/allocated_cc/results/<allocated_cc_result>/`, copy the three template
files into it, and replace the example name, description, and table contents.
The template CSV files contain headers only.

The `lcia_method` field must contain `ef31`. The aCC result therefore provides
all 16 EF v3.1 impact categories used for monitoring and target setting. PB-LCIA may still be
selected with the separate `lcia_method` argument of `run_optimization` to
characterize the selected plastics results.

The annual table contains every combination of case, world region, selected
impact category, and year from 2019 through 2100. Every value is finite,
strictly positive, and expressed in the matching method unit.
For `run_asr`, select completed LCA products containing the full `ef31` group;
use the same `lcia_method` selection as `run_lca`. ASRs use the EF v3.1 results.

For `run_optimization`, `optimization_lcia_method` must be `ef31`.
The separate `lcia_method` argument
selects the impacts calculated for the selected result.

In the article, the annual input represents steady-state allocated carrying
capacities. The cumulative input contains one GWP100 allocated carrying capacity
over 2019-2100 for each input label and world region. For the article's dynamic
climate change carrying capacity, pyaesa calculates the annual product of the allocated share
and the dynamic climate change carrying capacity within each Monte Carlo draw,
then sums those products over 2019-2100. It calculates each requested percentile
from the resulting cumulative distribution.

In the article, EF v3.1 GWP100 environmental burdens can include negative
contributions and emissions from agriculture, forestry and other land use
(AFOLU), whereas the dynamic climate change carrying capacity uses positive
Kyoto-gas emissions excluding AFOLU. This difference in accounting scope
remains part of the interpretation of the cumulative ASR.

The article uses P5, P50 and P95 percentiles of allocated carrying capacities.
Each percentile is used independently in Phase C. Each percentile supplies a deterministic allocated carrying capacity input to the ASR calculation.
The annual and cumulative inputs retain their distinct assessment periods.

`allocated_cc_case=None` selects every available input label independently.
The serialized field `allocated_cc_case` also accepts labels for other externally
supplied statistics or deterministic allocations. Record the allocation method, statistic, and data source represented by each
label.
