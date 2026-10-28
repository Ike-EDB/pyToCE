"""Inspect exact-candidate wheel and source-distribution archives."""

import argparse
import tarfile
import zipfile
from pathlib import Path

_SOURCE_ROOT_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    ".readthedocs.yaml",
    "MANIFEST.in",
    "pyproject.toml",
    "PKG-INFO",
    "setup.cfg",
}
_FORBIDDEN_CONTENT = (
    b"C:\\Users\\",
    b"/Users/",
    b"/home/runner/work/",
    b"-----BEGIN PRIVATE KEY-----",
    b"github_pat_",
)


def _one(directory: Path, pattern: str) -> Path:
    """Return the single distribution artifact matching ``pattern``."""
    matches = tuple(directory.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected one {pattern!r} artifact, found {matches!r}")
    return matches[0]


def _check_names(names: tuple[str, ...], *, wheel: bool) -> None:
    """Admit public archive roots, safe paths and the required tutorial inventory."""
    for name in names:
        parts = name.rstrip("/").split("/")
        if "\\" in name or ":" in parts[0] or any(p in {"", ".", ".."} for p in parts):
            raise AssertionError(f"Unsafe archive path: {name}")
        if not wheel:
            if not parts[0].startswith("pytoce-") or len(parts) < 2:
                raise AssertionError(f"Unexpected source archive root: {name}")
            parts = parts[1:]
        if any(p.startswith(".") for p in parts) and parts != [".readthedocs.yaml"]:
            raise AssertionError(f"Unexpected hidden archive path: {name}")
        if wheel:
            allowed = parts[0] == "pytoce" or (
                parts[0].startswith("pytoce-") and parts[0].endswith(".dist-info")
            )
        else:
            allowed = (
                (len(parts) == 1 and parts[0] in _SOURCE_ROOT_FILES)
                or parts[0] in {"docs", "tutorials"}
                or (
                    len(parts) >= 2
                    and parts[0] == "src"
                    and parts[1] in {"pytoce", "pytoce.egg-info"}
                )
            )
        if not allowed:
            raise AssertionError(f"Unexpected distribution member: {name}")
        lowered = "/" + name.lower()
        if (
            lowered.endswith((".7z", ".spold", ".spold.zip", ".zip"))
            or "/data_raw/ecoinvent/" in lowered
        ):
            raise AssertionError(f"Source or licensed archive leaked into artifact: {name}")
    notebook_count = sum(name.endswith(".ipynb") for name in names)
    if not any(("/" + name).endswith("/pytoce/py.typed") for name in names):
        raise AssertionError("Distribution is missing the installed typing marker")
    if wheel and notebook_count:
        raise AssertionError("Tutorial notebooks must not be installed by the wheel")
    if not wheel and notebook_count != 7:
        raise AssertionError(f"Source distribution has {notebook_count} tutorials, expected 7")


def _check_content(raw: bytes, *, member: str) -> None:
    """Reject machine-local paths and credentials in distributed content."""
    for marker in _FORBIDDEN_CONTENT:
        if marker in raw:
            raise AssertionError(f"Forbidden private content {marker!r} in {member}")


def verify(directory: Path) -> None:
    """Verify public artifact contents and leakage exclusions."""
    wheel = _one(directory, "*.whl")
    sdist = _one(directory, "*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        names = tuple(archive.namelist())
        _check_names(names, wheel=True)
        for name in names:
            _check_content(archive.read(name), member=name)
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = tuple(item for item in archive.getmembers() if item.isfile())
        if any(item.issym() or item.islnk() for item in archive.getmembers()):
            raise AssertionError("Source archive must not contain symbolic or hard links")
        names = tuple(item.name for item in members)
        _check_names(names, wheel=False)
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise AssertionError(f"Unreadable source-distribution member: {member.name}")
            _check_content(stream.read(), member=member.name)


def main() -> None:
    """Validate the wheel and source distribution in the requested directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_directory", type=Path)
    arguments = parser.parse_args()
    verify(arguments.artifact_directory.resolve())


if __name__ == "__main__":
    main()
