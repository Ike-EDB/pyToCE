"""Provision public REMIND-MFA 0.2.0 code before running package tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

from pytoce.download_data.sources.catalogue import load_catalogue, read_verified_source
from pytoce.download_data.sources.network import download_source
from pytoce.run_mfa.remind_mfa.execution import _extract_code


def main() -> None:
    """Download, authenticate and publish only the public source needed by tests."""
    repository = Path(__file__).resolve().parents[1]
    destination = repository / "build/test_dependencies/remind_mfa"
    if destination.resolve() != destination:
        raise ValueError(
            f"The test-source destination {destination} contains a filesystem link. "
            "Use an ordinary directory under this checkout's build directory."
        )
    if destination.exists():
        print(f"Reused REMIND-MFA 0.2.0 test source: {destination}")
        return

    sources, _ = load_catalogue()
    source = next(item for item in sources if item.source_id == "remind_mfa_code")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".remind_mfa-", dir=destination.parent, ignore_cleanup_errors=True
    ) as temporary:
        staging = Path(temporary)
        archive = staging / source.filename
        download_source(source, archive)
        read_verified_source(
            archive, expected_bytes=source.size_bytes, expected_sha256=source.sha256
        )
        extracted = _extract_code(archive, staging / "source")
        extracted.rename(destination)
    print(f"Prepared REMIND-MFA 0.2.0 test source: {destination}")


if __name__ == "__main__":
    main()
