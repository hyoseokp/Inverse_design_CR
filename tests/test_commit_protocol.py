from __future__ import annotations

from pathlib import Path

from crinv.commit_protocol import (
    CommitMarker,
    VALID_COMMITTED,
    VALID_NOT_RUN,
    VALID_STAGED,
)


def test_commit_marker_defaults_to_not_run(tmp_path: Path):
    m = CommitMarker(tmp_path / "valid.txt")
    assert m.read() == VALID_NOT_RUN


def test_commit_marker_stage_and_commit(tmp_path: Path):
    p = tmp_path / "valid.txt"
    m = CommitMarker(p)
    m.stage()
    assert m.read() == VALID_STAGED
    m.commit()
    assert m.read() == VALID_COMMITTED

