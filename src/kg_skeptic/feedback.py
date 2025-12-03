from __future__ import annotations

import json
import uuid
import datetime
from pathlib import Path
from typing import Literal


def append_claim_to_dataset(
    claim_text: str,
    evidence: list[str],
    label: Literal["PASS", "FAIL", "WARN"],
    output_path: str | Path = "data/annotated_claims.jsonl",
) -> str:
    """
    Appends a user-submitted claim with ground truth label to a JSONL file.

    Args:
        claim_text: The text of the claim.
        evidence: List of evidence strings (e.g. "PMID:12345").
        label: The user-provided ground truth label.
        output_path: Path to the JSONL file.

    Returns:
        The generated ID of the new record.
    """
    record_id = str(uuid.uuid4())

    # Structure evidence to be consistent with e2e_claim_fixtures.jsonl
    structured_evidence = []
    for ev in evidence:
        ev = ev.strip()
        if not ev:
            continue

        if ev.upper().startswith("PMID:"):
            structured_evidence.append({"type": "pubmed", "pmid": ev.split(":", 1)[1]})
        elif ev.upper().startswith("DOI:"):
            structured_evidence.append({"type": "doi", "id": ev.split(":", 1)[1]})
        else:
            structured_evidence.append({"type": "other", "id": ev})

    record = {
        "id": record_id,
        "claim": claim_text,
        "evidence": structured_evidence,
        "expected_decision": label,
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source": "user_feedback",
        },
    }

    path = Path(output_path)
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record_id
