"""Tests for NERVE models and JSON schema validation."""

import json
from pathlib import Path

import pytest

from nerve.models import (
    Claim,
    EntityMention,
    Finding,
    Report,
    Severity,
    SuggestedFix,
)
from nerve.schemas import REPORT_SCHEMA_PATH

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestEntityMention:
    def test_to_dict(self) -> None:
        entity = EntityMention(
            mention="TP53",
            norm_id="HGNC:11998",
            norm_label="tumor protein p53",
            source="dictionary",
        )
        d = entity.to_dict()
        assert d["mention"] == "TP53"
        assert d["norm_id"] == "HGNC:11998"
        assert d["source"] == "dictionary"

    def test_from_dict(self) -> None:
        data = {
            "mention": "BRCA1",
            "norm_id": "HGNC:1100",
            "norm_label": "BRCA1 DNA repair associated",
            "source": "dictionary",
            "metadata": {"type": "gene"},
        }
        entity = EntityMention.from_dict(data)
        assert entity.mention == "BRCA1"
        assert entity.norm_id == "HGNC:1100"
        assert entity.metadata == {"type": "gene"}


class TestClaim:
    def test_roundtrip(self) -> None:
        claim = Claim(
            id="claim-001",
            text="TP53 is a tumor suppressor.",
            entities=[EntityMention(mention="TP53", norm_id="HGNC:11998", source="dictionary")],
            evidence=["PMID:12345678"],
        )
        d = claim.to_dict()
        restored = Claim.from_dict(d)
        assert restored.id == claim.id
        assert restored.text == claim.text
        assert len(restored.entities) == 1
        assert restored.entities[0].mention == "TP53"


class TestFinding:
    def test_with_suggested_fix(self) -> None:
        fix = SuggestedFix(
            target_claim_id="claim-001",
            patch="Replace X with Y",
            rationale="X is incorrect",
            confidence=0.9,
        )
        finding = Finding(
            id="finding-001",
            kind="ontology_violation",
            severity=Severity.ERROR,
            message="Type mismatch detected",
            claim_id="claim-001",
            suggested_fix=fix,
        )
        d = finding.to_dict()
        assert d["severity"] == "error"
        suggested_fix = d["suggested_fix"]
        assert isinstance(suggested_fix, dict)
        assert suggested_fix["confidence"] == 0.9

        restored = Finding.from_dict(d)
        assert restored.severity == Severity.ERROR
        assert restored.suggested_fix is not None
        assert restored.suggested_fix.confidence == 0.9


class TestReport:
    def test_empty_report(self) -> None:
        report = Report(
            task_id="test-001",
            agent_name="test-agent",
            summary="No issues found",
        )
        d = report.to_dict()
        assert d["task_id"] == "test-001"
        assert d["claims"] == []
        assert d["findings"] == []

    def test_json_roundtrip(self) -> None:
        report = Report(
            task_id="test-002",
            agent_name="test-agent",
            summary="Found 1 issue",
            claims=[Claim(id="c1", text="Test claim", entities=[], evidence=["PMID:1"])],
            findings=[
                Finding(
                    id="f1",
                    kind="missing_evidence",
                    severity=Severity.WARNING,
                    message="No evidence",
                    claim_id="c1",
                )
            ],
            stats={"total_claims": 1, "total_findings": 1},
        )
        json_str = report.to_json()
        restored = Report.from_json(json_str)
        assert restored.task_id == report.task_id
        assert len(restored.claims) == 1
        assert len(restored.findings) == 1
        assert restored.findings[0].severity == Severity.WARNING

    def test_save_and_load(self, tmp_path: Path) -> None:
        report = Report(
            task_id="test-save-load",
            agent_name="test-agent",
            summary="Testing save and load",
        )
        report_path = tmp_path / "report.json"
        report.save(report_path)

        assert report_path.exists()

        loaded = Report.load(report_path)
        assert loaded.task_id == report.task_id
        assert loaded.agent_name == report.agent_name


class TestFixtures:
    """Test that all fixture files can be loaded and converted to Report objects."""

    @pytest.fixture
    def fixture_files(self) -> list[Path]:
        return list(FIXTURES_DIR.glob("report_*.json"))

    def test_fixtures_exist(self, fixture_files: list[Path]) -> None:
        assert len(fixture_files) >= 4, "Expected at least 4 fixture files"

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "report_clean.json",
            "report_ontology_violations.json",
            "report_missing_evidence.json",
            "report_contradictions.json",
        ],
    )
    def test_fixture_loads_as_report(self, fixture_name: str) -> None:
        fixture_path = FIXTURES_DIR / fixture_name
        with open(fixture_path) as f:
            data = json.load(f)

        report = Report.from_dict(data)
        assert report.task_id is not None
        assert report.agent_name is not None
        assert isinstance(report.claims, list)
        assert isinstance(report.findings, list)

    def test_clean_report_has_no_findings(self) -> None:
        with open(FIXTURES_DIR / "report_clean.json") as f:
            data = json.load(f)
        report = Report.from_dict(data)
        assert len(report.findings) == 0
        assert report.stats.get("total_findings") == 0

    def test_ontology_violations_report(self) -> None:
        with open(FIXTURES_DIR / "report_ontology_violations.json") as f:
            data = json.load(f)
        report = Report.from_dict(data)
        assert len(report.findings) == 3
        kinds = {f.kind for f in report.findings}
        assert "ontology_violation" in kinds
        assert "taxon_mismatch" in kinds
        assert "unnormalized_entity" in kinds

    def test_missing_evidence_report(self) -> None:
        with open(FIXTURES_DIR / "report_missing_evidence.json") as f:
            data = json.load(f)
        report = Report.from_dict(data)
        kinds = {f.kind for f in report.findings}
        assert "missing_evidence" in kinds
        blocker_findings = [f for f in report.findings if f.severity == Severity.BLOCKER]
        assert len(blocker_findings) >= 1

    def test_contradictions_report(self) -> None:
        with open(FIXTURES_DIR / "report_contradictions.json") as f:
            data = json.load(f)
        report = Report.from_dict(data)
        assert all(f.kind == "contradiction" for f in report.findings)


class TestSchemaExists:
    def test_schema_file_exists(self) -> None:
        assert REPORT_SCHEMA_PATH.exists()

    def test_schema_is_valid_json(self) -> None:
        with open(REPORT_SCHEMA_PATH) as f:
            schema = json.load(f)
        assert "$schema" in schema
        assert "properties" in schema
        assert "claims" in schema["properties"]
        assert "findings" in schema["properties"]
