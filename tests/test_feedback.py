"""Tests for feedback mechanism."""

import json
from pathlib import Path
from kg_skeptic.feedback import append_claim_to_dataset


def test_append_claim_to_dataset(tmp_path: Path) -> None:
    """Test appending a claim to the dataset."""
    output_path = tmp_path / "feedback.jsonl"

    # Test without comment
    record_id1 = append_claim_to_dataset(
        claim_text="Test claim 1",
        evidence=["PMID:12345"],
        label="PASS",
        output_path=output_path,
    )

    assert output_path.exists()

    with open(output_path) as f:
        lines = f.readlines()
        assert len(lines) == 1
        record1 = json.loads(lines[0])
        assert record1["id"] == record_id1
        assert record1["claim"] == "Test claim 1"
        assert record1["expected_decision"] == "PASS"
        assert "comment" not in record1["metadata"]

    # Test with comment
    record_id2 = append_claim_to_dataset(
        claim_text="Test claim 2",
        evidence=["PMID:67890"],
        label="FAIL",
        output_path=output_path,
        comment="This is a test comment",
    )

    with open(output_path) as f:
        lines = f.readlines()
        assert len(lines) == 2
        record2 = json.loads(lines[1])
        assert record2["id"] == record_id2
        assert record2["claim"] == "Test claim 2"
        assert record2["expected_decision"] == "FAIL"
        assert record2["metadata"]["comment"] == "This is a test comment"
