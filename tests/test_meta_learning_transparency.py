"""Tests for meta-learning transparency record writing."""

import json
from datetime import datetime, timezone

from polaris.abstractions.meta_learner import ProposalStatus
from polaris.core.meta_learning_transparency import MetaLearningTransparencyWriter


def test_meta_learning_transparency_writer_appends_jsonl(tmp_path):
    output_path = tmp_path / "meta-learning.jsonl"
    writer = MetaLearningTransparencyWriter(str(output_path))

    writer.record_cycle(
        {
            "cycle_id": "cycle-1",
            "system_id": "sys-a",
            "status": "completed",
            "recorded_at": datetime(2026, 2, 23, tzinfo=timezone.utc),
            "validation_status": ProposalStatus.APPROVED,
        }
    )
    writer.record_cycle(
        {
            "cycle_id": "cycle-2",
            "system_id": "sys-a",
            "status": "error",
        }
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["cycle_id"] == "cycle-1"
    assert rows[0]["recorded_at"] == "2026-02-23T00:00:00+00:00"
    assert rows[0]["validation_status"] == "approved"
    assert rows[1]["cycle_id"] == "cycle-2"


def test_meta_learning_transparency_writer_serializes_unknown_types(tmp_path):
    output_path = tmp_path / "meta-learning-unknowns.jsonl"
    writer = MetaLearningTransparencyWriter(str(output_path))

    class _Unknown:
        pass

    writer.record_cycle({"cycle_id": "cycle-1", "payload": _Unknown()})
    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["cycle_id"] == "cycle-1"
    assert "_Unknown" in row["payload"]
