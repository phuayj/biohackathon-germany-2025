"""Citation normalization utilities."""

from __future__ import annotations


def normalize_citation_identifier(identifier: str) -> str:
    """Normalize citation identifiers for matching across provenance and edges.

    Handles PMIDs, PMCs, and DOIs.

    Examples:
        - "12345" -> "PMID:12345"
        - "pmid:12345" -> "PMID:12345"
        - "PMCID:PMC12345" -> "PMC12345"
        - "doi:10.1000/1" -> "10.1000/1"
        - "https://doi.org/10.1000/1" -> "10.1000/1"
    """
    value = identifier.strip()
    if not value:
        return value

    lower = value.lower()

    # Normalize DOI URLs and prefixes to bare DOI strings.
    if lower.startswith("https://doi.org/") or lower.startswith("http://doi.org/"):
        value = value.split("doi.org/", 1)[-1]
        lower = value.lower()
    if lower.startswith("doi:"):
        value = value.split(":", 1)[-1].strip()
        lower = value.lower()

    upper = value.upper()

    # Normalize PMCID-style identifiers.
    if upper.startswith("PMCID:"):
        code = upper.split(":", 1)[-1].strip()
        if not code.startswith("PMC"):
            code = f"PMC{code}"
        return code
    if upper.startswith("PMC"):
        return upper

    # Normalize PMID-style identifiers.
    if upper.startswith("PMID:"):
        digits = upper.split(":", 1)[-1].strip()
        return f"PMID:{digits}"
    if value.isdigit():
        return f"PMID:{value}"

    # Bare DOIs starting with 10.* are returned as-is.
    if value.startswith("10."):
        return value

    return value
