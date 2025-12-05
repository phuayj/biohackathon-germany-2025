"""
COSMIC Cancer Gene Census MCP tool for gene function classification.

This adapter provides gene function lookup (oncogene vs tumor suppressor)
using the COSMIC Cancer Gene Census data. It is designed to support
semantic validation of biomedical claims involving gene causality.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from .provenance import ToolProvenance, make_live_provenance


class GeneFunctionClass(Enum):
    """Classification of gene function in cancer."""

    ONCOGENE = "oncogene"
    TUMOR_SUPPRESSOR = "tumor_suppressor"
    BOTH = "both"  # Can act as both oncogene and TSG
    FUSION = "fusion"  # Fusion partner only, no oncogene/TSG role
    UNKNOWN = "unknown"


@dataclass
class GeneFunction:
    """Gene function classification from COSMIC Cancer Gene Census."""

    gene_symbol: str
    function_class: GeneFunctionClass
    role_in_cancer_raw: str  # Original COSMIC ROLE_IN_CANCER value
    tier: int  # COSMIC tier (1 = strong evidence, 2 = less evidence)
    name: str  # Full gene name
    synonyms: list[str]
    provenance: ToolProvenance | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "gene_symbol": self.gene_symbol,
            "function_class": self.function_class.value,
            "role_in_cancer_raw": self.role_in_cancer_raw,
            "tier": self.tier,
            "name": self.name,
            "synonyms": self.synonyms,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @property
    def is_tumor_suppressor(self) -> bool:
        """Check if gene has tumor suppressor activity."""
        return self.function_class in (
            GeneFunctionClass.TUMOR_SUPPRESSOR,
            GeneFunctionClass.BOTH,
        )

    @property
    def is_oncogene(self) -> bool:
        """Check if gene has oncogene activity."""
        return self.function_class in (
            GeneFunctionClass.ONCOGENE,
            GeneFunctionClass.BOTH,
        )


def _parse_role_in_cancer(role: str) -> GeneFunctionClass:
    """Parse COSMIC ROLE_IN_CANCER field to GeneFunctionClass.

    COSMIC uses comma-separated values like:
    - "oncogene"
    - "TSG"
    - "oncogene, fusion"
    - "TSG, fusion"
    - "oncogene, TSG"
    - "oncogene, TSG, fusion"
    - "fusion"
    - "" (empty)
    """
    if not role or not role.strip():
        return GeneFunctionClass.UNKNOWN

    role_lower = role.lower()
    has_oncogene = "oncogene" in role_lower
    has_tsg = "tsg" in role_lower

    if has_oncogene and has_tsg:
        return GeneFunctionClass.BOTH
    if has_oncogene:
        return GeneFunctionClass.ONCOGENE
    if has_tsg:
        return GeneFunctionClass.TUMOR_SUPPRESSOR
    if "fusion" in role_lower:
        return GeneFunctionClass.FUSION

    return GeneFunctionClass.UNKNOWN


class COSMICTool:
    """MCP tool for accessing COSMIC Cancer Gene Census gene function data."""

    # Default path to the COSMIC CGC TSV file
    DEFAULT_CGC_PATH = Path(__file__).parent.parent.parent.parent / "data" / "cosmic"

    def __init__(self, cgc_path: Optional[Path] = None) -> None:
        """
        Initialize COSMIC tool.

        Args:
            cgc_path: Path to directory containing the COSMIC CGC TSV file.
                     Defaults to data/cosmic/ in the project root.
        """
        self.cgc_path = cgc_path or self.DEFAULT_CGC_PATH
        self._gene_index: dict[str, GeneFunction] | None = None
        self._synonym_index: dict[str, str] | None = None  # synonym -> gene_symbol

    def _find_cgc_file(self) -> Path | None:
        """Find the Cancer Gene Census TSV file."""
        if not self.cgc_path.exists():
            return None

        # Look for the TSV file (may have version in name)
        for f in self.cgc_path.glob("Cosmic_CancerGeneCensus_*.tsv"):
            return f

        # Also check for unversioned name
        simple_path = self.cgc_path / "cancer_gene_census.tsv"
        if simple_path.exists():
            return simple_path

        return None

    def _load_index(self) -> None:
        """Load the gene index from the CGC TSV file."""
        if self._gene_index is not None:
            return  # Already loaded

        self._gene_index = {}
        self._synonym_index = {}

        cgc_file = self._find_cgc_file()
        if cgc_file is None:
            return  # No file available

        with open(cgc_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                gene_symbol = row.get("GENE_SYMBOL", "").strip()
                if not gene_symbol:
                    continue

                role_raw = row.get("ROLE_IN_CANCER", "").strip()
                tier_raw = row.get("TIER", "2")
                name = row.get("NAME", "").strip()
                synonyms_raw = row.get("SYNONYMS", "").strip()

                # Parse tier
                try:
                    tier = int(tier_raw) if tier_raw else 2
                except ValueError:
                    tier = 2

                # Parse synonyms (comma-separated)
                synonyms: list[str] = []
                if synonyms_raw:
                    synonyms = [s.strip() for s in synonyms_raw.split(",") if s.strip()]

                function_class = _parse_role_in_cancer(role_raw)

                gene_func = GeneFunction(
                    gene_symbol=gene_symbol,
                    function_class=function_class,
                    role_in_cancer_raw=role_raw,
                    tier=tier,
                    name=name,
                    synonyms=synonyms,
                    provenance=make_live_provenance(
                        source_db="cosmic_cgc",
                        db_version="v103",
                    ),
                )

                self._gene_index[gene_symbol.upper()] = gene_func

                # Index synonyms
                for syn in synonyms:
                    syn_upper = syn.upper()
                    if syn_upper not in self._synonym_index:
                        self._synonym_index[syn_upper] = gene_symbol

    def lookup_gene(self, gene_symbol: str) -> GeneFunction | None:
        """
        Look up gene function classification by gene symbol.

        Args:
            gene_symbol: Gene symbol (e.g., "TP53", "BRCA1", "EGFR").
                        Case-insensitive.

        Returns:
            GeneFunction if found, None otherwise.
        """
        self._load_index()
        if self._gene_index is None:
            return None

        symbol_upper = gene_symbol.upper().strip()

        # Direct lookup
        if symbol_upper in self._gene_index:
            return self._gene_index[symbol_upper]

        # Try synonym lookup
        if self._synonym_index and symbol_upper in self._synonym_index:
            canonical = self._synonym_index[symbol_upper]
            return self._gene_index.get(canonical.upper())

        return None

    def get_function_class(self, gene_symbol: str) -> GeneFunctionClass:
        """
        Get the function class for a gene.

        Args:
            gene_symbol: Gene symbol (case-insensitive).

        Returns:
            GeneFunctionClass enum value.
        """
        gene = self.lookup_gene(gene_symbol)
        if gene is None:
            return GeneFunctionClass.UNKNOWN
        return gene.function_class

    def is_tumor_suppressor(self, gene_symbol: str) -> bool:
        """
        Check if a gene is a tumor suppressor.

        Args:
            gene_symbol: Gene symbol (case-insensitive).

        Returns:
            True if the gene has tumor suppressor activity (including genes
            that act as both oncogene and TSG).
        """
        gene = self.lookup_gene(gene_symbol)
        return gene.is_tumor_suppressor if gene else False

    def is_oncogene(self, gene_symbol: str) -> bool:
        """
        Check if a gene is an oncogene.

        Args:
            gene_symbol: Gene symbol (case-insensitive).

        Returns:
            True if the gene has oncogene activity (including genes
            that act as both oncogene and TSG).
        """
        gene = self.lookup_gene(gene_symbol)
        return gene.is_oncogene if gene else False

    def is_in_census(self, gene_symbol: str) -> bool:
        """
        Check if a gene is in the Cancer Gene Census.

        Args:
            gene_symbol: Gene symbol (case-insensitive).

        Returns:
            True if the gene is in the COSMIC Cancer Gene Census.
        """
        return self.lookup_gene(gene_symbol) is not None

    def get_all_tumor_suppressors(self) -> list[str]:
        """Get all tumor suppressor genes in the Census."""
        self._load_index()
        if self._gene_index is None:
            return []
        return [g.gene_symbol for g in self._gene_index.values() if g.is_tumor_suppressor]

    def get_all_oncogenes(self) -> list[str]:
        """Get all oncogenes in the Census."""
        self._load_index()
        if self._gene_index is None:
            return []
        return [g.gene_symbol for g in self._gene_index.values() if g.is_oncogene]
