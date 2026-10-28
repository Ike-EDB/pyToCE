"""Render user guides beside pyToCE workspace directories."""

import json
from importlib import resources
from pathlib import Path
from typing import cast

from .filesystem import atomic_write


def load_workspace_guide_catalogue(package: str) -> tuple[dict[str, object], ...]:
    """Load a workflow's packaged workspace guide catalog."""

    node = resources.files(package).joinpath("data", "workspace_guides.json")
    payload = json.loads(node.read_text(encoding="utf-8"))
    return tuple(cast(list[dict[str, object]], payload["guides"]))


def render_workspace_guide(package: str, guide_id: str) -> bytes:
    """Render one concise guide for a workspace directory."""

    guide = next(
        item for item in load_workspace_guide_catalogue(package) if item["guide_id"] == guide_id
    )

    def bullets(field: str) -> str:
        """Format one guide section as Markdown bullet lines."""
        return "\n".join(f"- {value}" for value in cast(list[str], guide[field]))

    expected = "\n".join(
        (
            "- `README.md`: guide to this directory's files and their use.",
            *(
                f"- `{item['name']}`: {item['description']}"
                for item in cast(list[dict[str, str]], guide["expected_contents"])
            ),
        )
    )
    provenance = ""
    if "provenance" in guide:
        provenance = f"\n## Provenance\n\n{bullets('provenance')}\n"
    text = f"""# {guide["title"]}

{guide["purpose"]}

Created by `{guide["created_by"]}`.

## Files

{expected}

## Inputs

{bullets("inputs")}

## Format

{bullets("tables_and_schemas")}
{bullets("units_signs_and_identifiers")}

## Read with Python

```python
{guide["pandas_example"]}
```

## Reuse

{bullets("reuse_and_refresh")}
{provenance}
## Notes

{bullets("scientific_limitations")}
"""
    return text.encode("utf-8")


def write_workspace_guide(
    directory: Path,
    package: str,
    guide_id: str,
) -> None:
    """Write one guide atomically beside the files it describes."""

    atomic_write(directory / "README.md", render_workspace_guide(package, guide_id))
