"""Tests for OWL-style ontology constraint checking."""

from __future__ import annotations


from nerve.ontology_constraints import (
    DLCheckResult,
    OntologyAxioms,
    OntologyConstraintChecker,
    check_dl_constraints,
)


class TestOntologyAxiomsLoading:
    """Test loading axioms from YAML."""

    def test_loads_default_axioms(self) -> None:
        axioms = OntologyAxioms.from_yaml()
        assert axioms.classes, "Should load class axioms"
        assert axioms.properties, "Should load property axioms"
        assert "gene" in axioms.classes
        assert "disease" in axioms.classes

    def test_class_disjointness_defined(self) -> None:
        axioms = OntologyAxioms.from_yaml()
        gene_axiom = axioms.classes.get("gene")
        assert gene_axiom is not None
        assert "disease" in gene_axiom.disjoint_with
        assert "phenotype" in gene_axiom.disjoint_with

    def test_property_domain_range_defined(self) -> None:
        axioms = OntologyAxioms.from_yaml()
        assoc_prop = axioms.properties.get("biolink:gene_associated_with_condition")
        assert assoc_prop is not None
        assert "gene" in assoc_prop.domain
        assert "disease" in assoc_prop.range or "phenotype" in assoc_prop.range


class TestDisjointnessChecking:
    """Test class disjointness constraint checking."""

    def test_gene_disease_disjoint(self) -> None:
        checker = OntologyConstraintChecker()
        (
            subj_viol,
            obj_viol,
            pair_viol,
            subj_pairs,
            obj_pairs,
            pair_pairs,
        ) = checker.check_disjointness("gene", "disease")

        assert pair_viol is True
        assert ("gene", "disease") in pair_pairs or ("disease", "gene") in pair_pairs

    def test_gene_pathway_not_disjoint(self) -> None:
        # Genes participate in pathways, so they should NOT be disjoint
        checker = OntologyConstraintChecker()
        (
            subj_viol,
            obj_viol,
            pair_viol,
            _,
            _,
            _,
        ) = checker.check_disjointness("gene", "pathway")

        assert pair_viol is False

    def test_same_category_not_disjoint(self) -> None:
        checker = OntologyConstraintChecker()
        (
            subj_viol,
            obj_viol,
            pair_viol,
            _,
            _,
            _,
        ) = checker.check_disjointness("gene", "gene")

        assert pair_viol is False


class TestDomainRangeChecking:
    """Test property domain/range constraint checking."""

    def test_gene_associated_with_condition_valid(self) -> None:
        checker = OntologyConstraintChecker()
        domain_valid, range_valid, _, _ = checker.check_domain_range(
            "biolink:gene_associated_with_condition", "gene", "disease"
        )
        assert domain_valid is True
        assert range_valid is True

    def test_gene_associated_with_condition_invalid_domain(self) -> None:
        checker = OntologyConstraintChecker()
        domain_valid, range_valid, expected_domain, _ = checker.check_domain_range(
            "biolink:gene_associated_with_condition", "disease", "phenotype"
        )
        assert domain_valid is False
        assert "gene" in expected_domain

    def test_expressed_in_valid(self) -> None:
        checker = OntologyConstraintChecker()
        domain_valid, range_valid, _, _ = checker.check_domain_range(
            "biolink:expressed_in", "gene", "tissue"
        )
        assert domain_valid is True
        assert range_valid is True

    def test_expressed_in_invalid_range(self) -> None:
        checker = OntologyConstraintChecker()
        domain_valid, range_valid, _, expected_range = checker.check_domain_range(
            "biolink:expressed_in", "gene", "disease"
        )
        assert domain_valid is True
        assert range_valid is False
        assert "tissue" in expected_range

    def test_unknown_predicate_passes(self) -> None:
        checker = OntologyConstraintChecker()
        domain_valid, range_valid, _, _ = checker.check_domain_range(
            "biolink:unknown_predicate", "gene", "disease"
        )
        assert domain_valid is True
        assert range_valid is True


class TestPropertyChainChecking:
    """Test property chain axiom checking."""

    def test_no_edges_no_chain(self) -> None:
        checker = OntologyConstraintChecker()
        support, paths, conflict, details = checker.check_property_chains(
            "biolink:expressed_in",
            "HGNC:1100",
            "gene",
            "UBERON:0000955",
            "tissue",
            kg_edges=None,
        )
        assert support is False
        assert conflict is False

    def test_chain_support_detected(self) -> None:
        checker = OntologyConstraintChecker()
        kg_edges = [
            {
                "subject": "HGNC:1100",
                "predicate": "biolink:participates_in",
                "object": "GO:0008150",
            },
            {
                "subject": "GO:0008150",
                "predicate": "biolink:located_in",
                "object": "UBERON:0000955",
            },
        ]
        support, paths, conflict, details = checker.check_property_chains(
            "biolink:expressed_in",
            "HGNC:1100",
            "gene",
            "UBERON:0000955",
            "tissue",
            kg_edges=kg_edges,
        )
        assert support is True
        assert len(paths) > 0


class TestRestrictionChecking:
    """Test existential and universal restriction checking."""

    def test_universal_restriction_violation(self) -> None:
        checker = OntologyConstraintChecker()
        univ_viol, univ_details, exist_viol, exist_details = checker.check_restrictions(
            "biolink:expressed_in", "gene", "disease", has_evidence=True
        )
        assert univ_viol is True
        assert univ_details.get("property") == "biolink:expressed_in"

    def test_universal_restriction_valid(self) -> None:
        checker = OntologyConstraintChecker()
        univ_viol, _, _, _ = checker.check_restrictions(
            "biolink:expressed_in", "gene", "tissue", has_evidence=True
        )
        assert univ_viol is False


class TestCheckAll:
    """Test the check_all method and convenience function."""

    def test_check_all_returns_result(self) -> None:
        checker = OntologyConstraintChecker()
        result = checker.check_all(
            predicate="biolink:gene_associated_with_condition",
            subject_id="HGNC:1100",
            subject_category="gene",
            object_id="MONDO:0005015",
            object_category="disease",
        )
        assert isinstance(result, DLCheckResult)
        assert result.domain_valid is True
        assert result.range_valid is True

    def test_check_dl_constraints_function(self) -> None:
        facts = check_dl_constraints(
            predicate="biolink:gene_associated_with_condition",
            subject_id="HGNC:1100",
            subject_category="gene",
            object_id="MONDO:0005015",
            object_category="disease",
        )
        assert isinstance(facts, dict)
        assert "domain_valid" in facts
        assert "range_valid" in facts
        assert "disjoint_pair_violation" in facts

    def test_to_facts_includes_all_keys(self) -> None:
        result = DLCheckResult()
        facts = result.to_facts()
        expected_keys = [
            "disjoint_subject_violation",
            "disjoint_object_violation",
            "disjoint_pair_violation",
            "domain_valid",
            "range_valid",
            "property_chain_support",
            "property_chain_conflict",
            "universal_violation",
            "existential_violation",
        ]
        for key in expected_keys:
            assert key in facts


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_axioms_permissive(self) -> None:
        checker = OntologyConstraintChecker(axioms=OntologyAxioms())
        result = checker.check_all(
            predicate="biolink:causes",
            subject_id="HGNC:1100",
            subject_category="gene",
            object_id="MONDO:0005015",
            object_category="disease",
        )
        assert result.domain_valid is True
        assert result.range_valid is True
        assert result.disjoint_pair_violation is False

    def test_case_insensitive_matching(self) -> None:
        checker = OntologyConstraintChecker()
        domain_valid, range_valid, _, _ = checker.check_domain_range(
            "biolink:gene_associated_with_condition", "Gene", "Disease"
        )
        assert domain_valid is True
        assert range_valid is True
