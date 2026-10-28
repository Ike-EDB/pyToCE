import json
from pathlib import Path

import pytest

from pytoce.run_optimization.runtime.candidate_results import (
    CandidateResultStore,
    CandidateResultStoreError,
    CandidateTask,
)
from pytoce.shared.runtime.file_lock import writer_lock


def _task(sequence: int = 0, *, candidate: str = "one") -> CandidateTask:
    return CandidateTask(
        sequence,
        f"candidate::{candidate}",
        json.dumps({"candidate": candidate}, sort_keys=True).encode(),
    )


def _open(parent: Path) -> CandidateResultStore:
    return CandidateResultStore.open(
        parent,
        candidate_input_identity="inputs::one",
        worker_authority_identity="worker::one",
    )


def test_candidate_result_is_saved_and_reused(tmp_path: Path) -> None:
    task = _task()
    with _open(tmp_path) as store:
        assert store.read_record(task.record) is None
        saved = store.save(task, b"result")
        assert saved == b"result"
        assert store.read_record(task.record) == saved

    with _open(tmp_path) as store:
        assert store.read_record(_task(8).record) == saved


def test_incomplete_candidate_cache_requires_explicit_removal(tmp_path: Path) -> None:
    """Reject missing completion manifests while preserving the incomplete payload."""
    empty_parent = tmp_path / "empty"
    (empty_parent / ".candidate_results").mkdir(parents=True)
    with pytest.raises(
        CandidateResultStoreError,
        match="manifest is missing.*Keep the cache unchanged",
    ):
        _open(empty_parent)

    row_parent = tmp_path / "row"
    with _open(row_parent) as store:
        row = store.directory / "00000000"
        row.mkdir()
        (row / "payload.bin").write_bytes(b"incomplete")
    with pytest.raises(
        CandidateResultStoreError,
        match="manifest is missing.*Keep the cache unchanged",
    ):
        _open(row_parent)
    assert (row / "payload.bin").read_bytes() == b"incomplete"


def test_candidate_cache_reports_an_existing_file_and_an_active_writer(tmp_path: Path) -> None:
    file_parent = tmp_path / "file"
    file_parent.mkdir()
    (file_parent / ".candidate_results").write_text("not a directory", encoding="utf-8")
    with pytest.raises(CandidateResultStoreError, match="cache is not a directory"):
        _open(file_parent)

    locked_parent = tmp_path / "locked"
    locked_parent.mkdir()
    with writer_lock(locked_parent / ".candidate_results.lock"):
        with pytest.raises(CandidateResultStoreError, match="Another package process"):
            _open(locked_parent)


def test_cache_reuses_interventions_after_search_order_changes(tmp_path: Path) -> None:
    """Insert finer-search endpoints and preserve saved results."""
    with _open(tmp_path) as store:
        store.save(_task(), b"first-result")
        store.save(_task(1, candidate="two"), b"second-result")
    original = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and path.suffix != ".lock"
    }

    with _open(tmp_path) as store:
        assert store.read_record(_task(0, candidate="new").record) is None
        store.save(_task(0, candidate="new"), b"new-result")
        assert store.read_record(_task(1, candidate="two").record) == b"second-result"
        assert store.read_record(_task(2).record) == b"first-result"
    assert all((tmp_path / path).read_bytes() == content for path, content in original.items())
    with _open(tmp_path) as store:
        assert store.read_record(_task(7, candidate="new").record) == b"new-result"


@pytest.mark.parametrize(
    ("candidate_input_identity", "worker_identity"),
    (("inputs::two", "worker::one"), ("inputs::one", "worker::two")),
)
def test_candidate_cache_rejects_different_request_inputs(
    tmp_path: Path, candidate_input_identity: str, worker_identity: str
) -> None:
    with _open(tmp_path):
        pass

    with pytest.raises(CandidateResultStoreError, match="belongs to different inputs"):
        CandidateResultStore.open(
            tmp_path,
            candidate_input_identity=candidate_input_identity,
            worker_authority_identity=worker_identity,
        )


def test_candidate_cache_rejects_a_non_directory_row(tmp_path: Path) -> None:
    with _open(tmp_path) as store:
        row = store.directory / "00000000"
        row.write_text("not a directory", encoding="utf-8")
    with pytest.raises(CandidateResultStoreError, match="is not a directory"):
        _open(tmp_path)


def test_old_request_owned_cache_is_preserved_and_not_adopted(tmp_path: Path) -> None:
    """An earlier owner contract cannot silently become a reusable calculation cache."""
    directory = tmp_path / ".candidate_results"
    directory.mkdir()
    content = json.dumps(
        {
            "schema_version": "pytoce_candidate_result_owner_schema_v1",
            "request_identity": "request::one",
            "worker_authority_identity": "worker::one",
        }
    ).encode()
    manifest = directory / "manifest.json"
    manifest.write_bytes(content)
    with pytest.raises(CandidateResultStoreError, match="belongs to different inputs"):
        _open(tmp_path)
    assert manifest.read_bytes() == content
