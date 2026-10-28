# Package architecture

The package root lazily exposes seven functions:

```text
set_workspace
download_data
prepare_backgrounds
run_mfa
run_lca
run_asr
run_optimization
```

Each public function is implemented by the package with the same name.
Its coordinator owns the public arguments and returned report. The package root
loads public functions lazily when their attributes are first accessed.

Package data is stored below the workflow that uses it. Subpackage
`__init__.py` files contain documentation only, and internal modules import
objects from the module that defines them.

`shared/runtime` contains the small set of filesystem and Brightway operations
used by more than one workflow. `shared/assessment` owns the ASR calculations
used by monitoring and target setting. Other scientific calculations remain
with their workflow owners.

The workflow dependencies are:

```text
set_workspace -> download_data -> run_mfa
download_data -> prepare_backgrounds
completed MFA result + prepared background -> run_lca
completed LCA result + allocated carrying capacity input -> run_asr
allocated carrying capacity input + prepared backgrounds
    + required baseline LCA results + completed foreground MFA result
    -> run_optimization
```

`run_optimization` reads these completed parent inputs and calculates each
candidate's MFA trajectory internally. Each selected foreground uses its
own completed MFA and baseline LCA results, supplied through the preceding
public workflows.
