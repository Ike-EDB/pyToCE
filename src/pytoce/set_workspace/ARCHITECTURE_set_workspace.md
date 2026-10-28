# `set_workspace` architecture

`set_workspace.py` resolves the selected directory, creates `data_raw`,
`data_processed`, and `results`, selects the workspace for the current Python
session, and returns `WorkspaceReport`.

`runtime/installation.py` installs package templates, examples, schemas,
guides, and methodological notes. With `refresh=True`, it restores only those
package-owned files. It preserves user inputs with other filenames and all
scientific results.

`runtime/workspace.py` stores the active path for the current Python session.
The other public workflows obtain the path from this owner.
