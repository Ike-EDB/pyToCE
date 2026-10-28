# Foreground scenario YAML guide

This directory contains non-runnable templates and focused examples for
user-authored plastics foreground scenarios.

## Files in this directory

Runnable YAML belongs only in the sibling `scenarios/` directory. This
directory contains `foreground_scenario.yaml`, `examples/`, this guide, and the
readable `default_annual_input_values.csv` lookup. The Parquet defaults and JSON
parameter catalog are in
`../default_inputs/default_input_values.parquet` and
`../default_inputs/parameter_catalogue.json`. The runnable YAML grammar is
`../schemas/foreground_scenario_schema_v1.json`.

The package reads the selected coordinates, expands annual trajectories, and
checks bounds and share totals. `run_mfa` calculates stocks, flows, trade,
treatment, and the nominal carbon account. Constituent carbon custody and
LCI carbon remain separate quantities.

## Quick start

1. Copy `foreground_scenario.yaml` or one focused example.
2. Give the file a unique lowercase `snake_case` stem and put it in
   `scenarios/`.
3. Set `foreground_scenario` equal to that stem.
4. Keep `schema_version: pytoce_foreground_scenario_schema_v1` and
   `ssp_scenario: SSP2`.
5. Inspect default foreground parameter values and units before choosing values.
6. Run `run_mfa(foreground_scenario="your_name")`. If you intentionally edit
   the same scenario, use `run_mfa(foreground_scenario="your_name", refresh=True)`.

Scenario changes begin in 2020 and retain history through 2019.
Omitted coordinates retain the default foreground parameter value for that year.
A value applies to each listed coordinate or to all coordinates
chosen by a named broad selector.

## YAML conventions and grammar

Each entry under `changes` identifies an individual `parameter` or a
`parameter_group`. An individual parameter accepts dated `changes` or a
supported `trajectory`. A parameter group accepts only a `linear_from_2019`
trajectory with `selector: uniform_all_h12`. Duplicate keys, overlapping rules,
unknown fields, and incomplete groups are invalid.

```yaml
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: user_case
ssp_scenario: SSP2
changes:
  - parameter: collection_rate
    region: EUR
    changes:
      - year: 2030
        mode: absolute
        value: 0.70
        interpolation: step
```

Use a singular selector for one coordinate and a plural selector for an
explicit nonempty list. Broad selectors include `uniform_all_h12`,
`uniform_all_goods`, and `uniform_all_h12_goods`. Absolute mass values require explicit coordinates.

## Annual change points and modes

Ordinary change points use step changes. `interpolation` may be omitted or set
to `step`. Use `trajectory` for a linear change from 2019.

| Mode | Meaning |
|---|---|
| `absolute` | Set the selected coordinate to the stated value. |
| `retain_default` | Follow the default for each year. |
| `multiplier_from_default` | Multiply the default for each year. |
| `delta_from_default` | Add to the default for each year. |
| `retain_2019` | Hold the selected coordinate at its 2019 value. |
| `multiplier_from_2019` | Multiply the selected coordinate's 2019 value. |
| `delta_from_2019` | Add to the selected coordinate's 2019 value. |
| `retain_2019_propensity_pretrade_collected` | For waste trade, keep the 2019 trade share and apply it to current collected waste. |
| `retain_default_propensity_pretrade_collected` | For waste trade, apply each year's default trade share to current collected waste. |

## Linear trajectories

A trajectory begins from each selected coordinate's 2019 value,
reaches its target in an integer year from 2020 through 2100, and persists.

```yaml
- parameter: collection_rate
  selector: uniform_all_h12
  trajectory:
    mode: linear_from_2019
    target_year: 2050
    target_mode: absolute
    target_value: 0.80
```

Scalar `target_mode` is `absolute`, `multiplier_from_2019`, or
`delta_from_2019`. The annual value is
`start_2019 + clip((year-2019)/(target_year-2019), 0, 1) * (target-start_2019)`.

Treatment and virgin feedstock groups require every share:

```yaml
- parameter_group: collected_waste_treatment_shares
  selector: uniform_all_h12
  trajectory:
    mode: linear_from_2019
    target_year: 2050
    target_values:
      mechanical_recycling_rate: 0.40
      chemical_recycling_rate: 0.10
      incineration_rate: 0.30
      landfill_share: 0.20

- parameter_group: virgin_feedstock_shares
  selector: uniform_all_h12
  trajectory:
    mode: linear_from_2019
    target_year: 2050
    target_values:
      fossil_share: 0.70
      bio_production_rate: 0.20
      daccu_production_rate: 0.10
```

Every share is required, finite, and nonnegative, and the shares must sum to
one. Linear interpolation
preserves the annual total.
`trajectory` and ordinary `changes` are mutually exclusive. Waste trade,
`plastic_family_share`, derived outputs, discrete parameters, and incompatible
selectors cannot use a trajectory. Material-share changes use dated change
points and must supply all twelve plastic families at each changed
region/good/year.

## Editable parameter reference

| Parameter | Meaning and main coordinates |
|---|---|
| `stock_projection` | prospective in-use stock target by year, world region, and good |
| `lifetime_mean` | mean product lifetime by year and good |
| `collection_rate` | share of actual-year discard collected by year and world region |
| `mechanical_recycling_rate` | mechanical share of collected waste after waste trade by year and world region |
| `chemical_recycling_rate` | chemical share of collected waste after waste trade by year and world region |
| `incineration_rate` | incineration share of collected waste after waste trade by year and world region |
| `plastic_family_share` | complete material share of new inflow by year, region, material, and good |
| `waste_imports` | source waste-import mass or deferred propensity by year and region |
| `waste_exports` | source waste-export mass or deferred propensity by year and region |
| `bio_production_rate` | biogenic virgin-feed production route share by year and region |
| `daccu_production_rate` | DACCU virgin-feed production route share by year and region |
| `population` | SSP2 population driver by year and world region |
| `gdppc` | SSP2 GDP-per-capita driver by year and world region |

`landfill_share` and `fossil_share` are explicit group members and derived
residuals in the upstream interface. An individual absolute `landfill_share`
value only asserts the residual left by the active mechanical-recycling,
chemical-recycling and incineration controls. A treatment-group trajectory supplies all four shares, while
compilation retains the three independent treatment controls. Reported landfill
share is derived from realized direct treatment quantities.

Change `lifetime_mean` only. When the mean lifetime is changed, the standard
deviation is changed proportionally to retain the coefficient of variation of
the original lifetime distributions. The package derives the standard deviation
in years as `default_std * (mean / default_mean)` for each end-use group and year.
Thus a 50% longer mean also gives a 50% larger standard deviation.
The compiled table records `lifetime_std` as a derived input.

## How changes affect the MFA

Choose direct stock targets or socioeconomic drivers within one scenario:
a scenario uses either `stock_projection` or the `population` and `gdppc` drivers.

History and the end-2019 opening state remain fixed. Lifetime transitions
preserve surviving cohort amount, entry year, age, material, carbon origin,
region, and good. Each year's lifetime distribution conditions the survival of
both existing and new products. Its mean defines the updated lifetime distribution. The remaining lifespan of
each existing cohort follows from its age and conditional survival under that
distribution. Lifetime extension is represented by changes in the lifetime distribution. Material-share changes affect new inflow only. Operational
controls apply in the activity year.

The article calls the included `frozen_2019_default_baseline`
2019frozen_SSP2demand. It follows the SSP2-driven default after 2019 for stock
demand, lifetimes, end-use material composition, waste-trade propensities,
and all other non-controls. It freezes each
region's 2019 collection, treatment shares, and virgin feedstock shares. Annual
waste imports and exports are calculated from the default trade shares and that
year's collection. Mechanical recovery and its air, water and managed solid
losses follow fixed constituent recipes. The package allocates recycling inputs
under constituent compatibility and receiving-demand limits.

## Defaults, calculated values, and reuse

```python
import pandas as pd

defaults = pd.read_csv(
    workspace / "data_raw/foreground_scenarios/templates/default_annual_input_values.csv"
)
```

The compiled table records the value and source used for each coordinate.
`run_mfa` calculates stock inflow and outflow, discard, collected mass, virgin
production, recycling, landfill, and fossil share.

A completed result is reused when its manifest is present. After changing a
scenario file under the same name, call
`run_mfa(..., refresh=True)` or choose a new scenario name. Refresh replaces
that scenario's MFA result. To compare alternatives, keep each scenario and
result under its own name.

## Focused examples

- [Stock volume](examples/01_stock_volume.yaml)
- [Lifetime transitions](examples/02_lifetime.yaml)
- [Plastic-family mix](examples/03_plastic_family_mix.yaml)
- [Circularity and treatment](examples/04_circularity_treatment.yaml)
- [Virgin feedstock](examples/05_virgin_feedstock.yaml)
- [Waste trade](examples/06_waste_trade.yaml)
- [Socioeconomic drivers](examples/07_socioeconomic_drivers.yaml)

## Example syntax

| Syntax | Example |
|---|---|
| `stock_projection`; default-relative modes; `uniform_all_h12_goods` | stock volume |
| `lifetime_mean`; `good`, `goods`; `linear_from_2019` | lifetime transitions |
| `plastic_family_share`; `region`, `materials` | plastic-family mix |
| `collection_rate`; `collected_waste_treatment_shares`; `mechanical_recycling_rate`, `chemical_recycling_rate`, `incineration_rate`, `landfill_share` | circularity and treatment |
| `virgin_feedstock_shares`; `fossil_share`, `bio_production_rate`, `daccu_production_rate` | virgin feedstock |
| `waste_imports`, `waste_exports`; `regions`; propensity mode | waste trade |
| `population`, `gdppc`; singular/plural regions | socioeconomic drivers |

The examples and skeleton show `uniform_all_h12` and
`uniform_all_h12_goods`, plus explicit `region`, `regions`, `materials`, `good`,
and `goods` syntax. Use `material` when specifying one material; a material-share
change must still provide the complete partition through its other entries.
For lifetime mean, `uniform_all_goods` applies the same change to every end-use group.

## Scientific limitations

The package checks the YAML structure before running MFA; `run_mfa` then checks
stock and flow feasibility under the cohort, source-trade and recovery
assumptions. The scenario specifies model inputs. The MFA checks material-flow feasibility,
and the LCA calculates the resulting environmental burdens. Bio and DACCU targets
select virgin-feed production routes. Ultimate polymer carbon origin also
depends on constituent chemistry and recycled-material custody. The article
describes the material and provider assumptions.
