"""Exercise public artifact membership and credential rejection."""

import pytest
from verify_release_artifacts import _check_content, _check_names

_WHEEL = ("pytoce/py.typed", "pytoce-1.0.0.dist-info/METADATA")
_SDIST = (
    "pytoce-1.0.0/src/pytoce/py.typed",
    "pytoce-1.0.0/.readthedocs.yaml",
    *(f"pytoce-1.0.0/tutorials/example_{index}.ipynb" for index in range(7)),
)


@pytest.mark.parametrize(("members", "wheel"), ((_WHEEL, True), (_SDIST, False)))
def test_public_distribution_members_are_admitted(members: tuple[str, ...], wheel: bool) -> None:
    """Accept both documented layouts, including the source-only notebook inventory."""
    _check_names(members, wheel=wheel)


@pytest.mark.parametrize(
    ("member", "wheel"),
    (
        ("notes/review.md", True),
        ("pytoce-1.0.0/notes/review.md", False),
        ("pytoce/.local/notes.txt", True),
        ("pytoce-1.0.0/docs/.local/notes.txt", False),
        ("pytoce/data/source.7z", True),
        ("pytoce-1.0.0/src/pytoce/data/source.spold", False),
        ("pytoce/data_raw/ecoinvent/inventory.json", True),
        ("/pytoce/absolute.py", True),
        ("pytoce/../outside.py", True),
        ("C:/pytoce/absolute.py", True),
        ("pytoce\\module.py", True),
        ("other-1.0.0/src/pytoce/module.py", False),
    ),
)
def test_unexpected_or_unsafe_members_are_rejected(member: str, wheel: bool) -> None:
    """Reject unintended content and unsafe member names before reading archives."""
    members = _WHEEL if wheel else _SDIST
    with pytest.raises(AssertionError):
        _check_names((*members, member), wheel=wheel)


@pytest.mark.parametrize(
    "content",
    (
        b"C:\\Users\\example",
        b"/Users/example",
        b"-----BEGIN PRIVATE KEY-----",
        b"github_pat_example",
    ),
)
def test_machine_paths_and_credentials_are_rejected(content: bytes) -> None:
    """Keep the portable-path and secret-content checks effective."""
    with pytest.raises(AssertionError):
        _check_content(content, member="example.txt")


def test_ordinary_public_text_is_admitted() -> None:
    """Retain ordinary documentation and relative paths."""
    _check_content(
        b"Article and Methodological appendix; data_raw/foreground_scenarios/", member="example.txt"
    )
