import json
from pathlib import Path

import numpy as np
import pandas as pd

from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.set_workspace.set_workspace import set_workspace

_BASELINE = "frozen_2019_default_baseline"
_COORDINATES = ["region", "material", "good", "process", "route"]


def test_default_baseline_follows_default_non_controls_and_freezes_share_totals(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    report = set_workspace(root)
    assert (report.foreground_scenarios / f"{_BASELINE}.yaml").is_file()
    assert not (
        report.foreground_scenarios / "frozen_2019_circularity_2019_end_use_material_mix.yaml"
    ).exists()

    compiled = compile_foreground_scenario(root, _BASELINE)
    table = pd.read_parquet(compiled.compiled_parameters_path)
    defaults = pd.read_parquet(report.default_input_parameters)

    non_controls = {
        "stock_projection",
        "lifetime_mean",
        "lifetime_std",
        "plastic_family_share",
    }
    followed = table.loc[table["parameter_id"].isin(non_controls)]
    assert set(followed["value_mode"]) == {"retain_default"}
    assert followed["resolved_value"].equals(followed["default_value"])

    frozen_controls = {
        "collection_rate",
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "bio_production_rate",
        "daccu_production_rate",
    }
    packaged_2019 = defaults.loc[
        defaults["year"].eq(2019) & defaults["parameter_id"].isin(frozen_controls),
        ["parameter_id", *_COORDINATES, "default_value"],
    ].rename(columns={"default_value": "expected_2019"})
    frozen = table.loc[table["parameter_id"].isin(frozen_controls)].merge(
        packaged_2019,
        on=["parameter_id", *_COORDINATES],
        validate="many_to_one",
    )
    assert set(frozen["value_mode"]) == {"retain_2019"}
    assert np.array_equal(frozen["resolved_value"], frozen["expected_2019"])
    assert np.array_equal(frozen["reference_value"], frozen["expected_2019"])

    rates_2019 = defaults.loc[
        defaults["year"].eq(2019)
        & defaults["parameter_id"].isin(
            {
                "mechanical_recycling_rate",
                "chemical_recycling_rate",
                "incineration_rate",
                "bio_production_rate",
                "daccu_production_rate",
            }
        )
    ].pivot(index="region", columns="parameter_id", values="default_value")
    treatment = rates_2019[
        [
            "mechanical_recycling_rate",
            "chemical_recycling_rate",
            "incineration_rate",
        ]
    ].sum(axis=1)
    feedstock = rates_2019[["bio_production_rate", "daccu_production_rate"]].sum(axis=1)
    assert (1.0 - treatment >= 0.0).all()
    assert (1.0 - feedstock >= 0.0).all()
    assert np.allclose(treatment + (1.0 - treatment), 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(feedstock + (1.0 - feedstock), 1.0, rtol=0.0, atol=1e-12)

    waste_trade = table.loc[table["parameter_id"].isin({"waste_imports", "waste_exports"})]
    assert set(waste_trade["value_mode"]) == {"retain_default_propensity_pretrade_collected"}
    assert waste_trade["resolved_value"].isna().all()
    assert waste_trade["deferred_propensity"].isna().all()
    assert set(waste_trade["deferred_denominator_id"]) == {"local_pretrade_collected_waste"}

    manifest = json.loads(compiled.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "pytoce_compiled_foreground_scenario_schema_v1"
    assert manifest["normalized_scenario"]["trajectories"] == []
