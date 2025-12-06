"""OWL-style ontology constraint checking for NERVE.

This module evaluates Description Logic (DL) axioms against biomedical claims:
- Class disjointness (A ⊓ B = ⊥)
- Property domain/range constraints
- Property chains (P ∘ Q ⊑ R)
- Existential/universal restrictions

Axioms are loaded from ontology_axioms.yaml and evaluated to produce facts
that feed into the rule engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import yaml

DEFAULT_AXIOMS_PATH = Path(__file__).resolve().parents[2] / "ontology_axioms.yaml"


@dataclass
class ClassAxiom:
    """Class definition with disjointness constraints."""

    name: str
    description: str = ""
    iri_prefixes: list[str] = field(default_factory=list)
    disjoint_with: list[str] = field(default_factory=list)


@dataclass
class PropertyAxiom:
    """Object property with domain/range constraints."""

    predicate: str
    description: str = ""
    domain: list[str] = field(default_factory=list)
    range: list[str] = field(default_factory=list)


@dataclass
class PropertyChainAxiom:
    """Property chain axiom (P ∘ Q ⊑ R)."""

    id: str
    description: str = ""
    chain: list[dict[str, str]] = field(default_factory=list)
    inferred_predicate: str = ""
    inferred_subject_type: str = ""
    inferred_object_type: str = ""


@dataclass
class RestrictionAxiom:
    """Existential or universal restriction."""

    id: str
    kind: str  # "existential" or "universal"
    description: str = ""
    on_class: str = ""
    on_property: str = ""
    property: str = ""
    domain: str = ""
    filler: str = ""
    allowed_fillers: list[str] = field(default_factory=list)
    severity: str = "warn"
    requires_evidence: bool = False


@dataclass
class OntologyAxioms:
    """Container for all ontology axioms."""

    classes: dict[str, ClassAxiom] = field(default_factory=dict)
    properties: dict[str, PropertyAxiom] = field(default_factory=dict)
    chains: list[PropertyChainAxiom] = field(default_factory=list)
    restrictions: list[RestrictionAxiom] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "OntologyAxioms":
        """Load axioms from YAML file."""
        axioms_path = Path(path) if path else DEFAULT_AXIOMS_PATH
        if not axioms_path.exists():
            return cls()

        with axioms_path.open() as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            return cls()

        classes: dict[str, ClassAxiom] = {}
        raw_classes = raw.get("classes", {})
        if isinstance(raw_classes, dict):
            for name, data in raw_classes.items():
                if isinstance(data, dict):
                    classes[str(name)] = ClassAxiom(
                        name=str(name),
                        description=str(data.get("description", "")),
                        iri_prefixes=_as_str_list(data.get("iri_prefixes")),
                        disjoint_with=_as_str_list(data.get("disjoint_with")),
                    )

        properties: dict[str, PropertyAxiom] = {}
        raw_props = raw.get("object_properties", {})
        if isinstance(raw_props, dict):
            for pred, data in raw_props.items():
                if isinstance(data, dict):
                    properties[str(pred)] = PropertyAxiom(
                        predicate=str(pred),
                        description=str(data.get("description", "")),
                        domain=_as_str_list(data.get("domain")),
                        range=_as_str_list(data.get("range")),
                    )

        chains: list[PropertyChainAxiom] = []
        raw_chains = raw.get("property_chains", [])
        if isinstance(raw_chains, list):
            for item in raw_chains:
                if isinstance(item, dict):
                    chain_steps = item.get("chain", [])
                    if isinstance(chain_steps, list):
                        chains.append(
                            PropertyChainAxiom(
                                id=str(item.get("id", "")),
                                description=str(item.get("description", "")),
                                chain=[
                                    {str(k): str(v) for k, v in step.items()}
                                    for step in chain_steps
                                    if isinstance(step, dict)
                                ],
                                inferred_predicate=str(item.get("inferred_predicate", "")),
                                inferred_subject_type=str(item.get("inferred_subject_type", "")),
                                inferred_object_type=str(item.get("inferred_object_type", "")),
                            )
                        )

        restrictions: list[RestrictionAxiom] = []
        raw_restrictions = raw.get("restrictions", [])
        if isinstance(raw_restrictions, list):
            for item in raw_restrictions:
                if isinstance(item, dict):
                    restrictions.append(
                        RestrictionAxiom(
                            id=str(item.get("id", "")),
                            kind=str(item.get("kind", "existential")),
                            description=str(item.get("description", "")),
                            on_class=str(item.get("on_class", "")),
                            on_property=str(item.get("on_property", "")),
                            property=str(item.get("property", "")),
                            domain=str(item.get("domain", "")),
                            filler=str(item.get("filler", "")),
                            allowed_fillers=_as_str_list(item.get("allowed_fillers")),
                            severity=str(item.get("severity", "warn")),
                            requires_evidence=bool(item.get("requires_evidence", False)),
                        )
                    )

        return cls(
            classes=classes,
            properties=properties,
            chains=chains,
            restrictions=restrictions,
        )


def _as_str_list(value: object) -> list[str]:
    """Convert value to list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


@dataclass
class DLCheckResult:
    """Result of DL constraint checking."""

    disjoint_subject_violation: bool = False
    disjoint_object_violation: bool = False
    disjoint_pair_violation: bool = False
    disjoint_subject_types: list[tuple[str, str]] = field(default_factory=list)
    disjoint_object_types: list[tuple[str, str]] = field(default_factory=list)
    disjoint_pair_types: list[tuple[str, str]] = field(default_factory=list)

    domain_valid: bool = True
    range_valid: bool = True
    domain_expected: list[str] = field(default_factory=list)
    range_expected: list[str] = field(default_factory=list)

    property_chain_support: bool = False
    property_chain_paths: list[dict[str, object]] = field(default_factory=list)
    property_chain_conflict: bool = False
    property_chain_conflict_details: dict[str, object] = field(default_factory=dict)

    universal_violation: bool = False
    universal_violation_details: dict[str, object] = field(default_factory=dict)
    existential_violation: bool = False
    existential_violation_details: dict[str, object] = field(default_factory=dict)

    def to_facts(self) -> dict[str, object]:
        """Convert to facts dictionary for rule engine."""
        facts: dict[str, object] = {
            "disjoint_subject_violation": self.disjoint_subject_violation,
            "disjoint_object_violation": self.disjoint_object_violation,
            "disjoint_pair_violation": self.disjoint_pair_violation,
            "domain_valid": self.domain_valid,
            "range_valid": self.range_valid,
            "property_chain_support": self.property_chain_support,
            "property_chain_conflict": self.property_chain_conflict,
            "universal_violation": self.universal_violation,
            "existential_violation": self.existential_violation,
        }

        if self.disjoint_subject_types:
            facts["disjoint_subject_types"] = [f"{a} ⊓ {b}" for a, b in self.disjoint_subject_types]
        if self.disjoint_object_types:
            facts["disjoint_object_types"] = [f"{a} ⊓ {b}" for a, b in self.disjoint_object_types]
        if self.disjoint_pair_types:
            facts["disjoint_pair_types"] = [f"{a} ⊓ {b}" for a, b in self.disjoint_pair_types]

        if self.domain_expected:
            facts["domain_expected"] = self.domain_expected
        if self.range_expected:
            facts["range_expected"] = self.range_expected

        if self.property_chain_paths:
            facts["property_chain_paths"] = self.property_chain_paths
        if self.property_chain_conflict_details:
            facts["property_chain_conflict_details"] = self.property_chain_conflict_details

        if self.universal_violation_details:
            facts["universal_violation_details"] = self.universal_violation_details
        if self.existential_violation_details:
            facts["existential_violation_details"] = self.existential_violation_details

        return facts


class OntologyConstraintChecker:
    """Evaluates OWL-style constraints against claims."""

    def __init__(self, axioms: OntologyAxioms | None = None) -> None:
        self.axioms = axioms or OntologyAxioms.from_yaml()

    def check_disjointness(
        self,
        subject_category: str,
        object_category: str,
    ) -> tuple[
        bool, bool, bool, list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]
    ]:
        """Check class disjointness constraints.

        Returns:
            Tuple of (subject_violation, object_violation, pair_violation,
                     subject_pairs, object_pairs, pair_pairs)
        """
        subject_violation = False
        object_violation = False
        pair_violation = False
        subject_pairs: list[tuple[str, str]] = []
        object_pairs: list[tuple[str, str]] = []
        pair_pairs: list[tuple[str, str]] = []

        subj_lower = subject_category.lower()
        obj_lower = object_category.lower()

        subj_axiom = self.axioms.classes.get(subj_lower)
        obj_axiom = self.axioms.classes.get(obj_lower)

        if subj_axiom and obj_lower in subj_axiom.disjoint_with:
            pair_violation = True
            pair_pairs.append((subj_lower, obj_lower))

        if obj_axiom and subj_lower in obj_axiom.disjoint_with:
            if not pair_violation:
                pair_violation = True
                pair_pairs.append((obj_lower, subj_lower))

        return (
            subject_violation,
            object_violation,
            pair_violation,
            subject_pairs,
            object_pairs,
            pair_pairs,
        )

    def check_domain_range(
        self,
        predicate: str,
        subject_category: str,
        object_category: str,
    ) -> tuple[bool, bool, list[str], list[str]]:
        """Check property domain/range constraints.

        Returns:
            Tuple of (domain_valid, range_valid, expected_domain, expected_range)
        """
        pred_lower = predicate.lower()
        subj_lower = subject_category.lower()
        obj_lower = object_category.lower()

        prop_axiom = self.axioms.properties.get(predicate)
        if prop_axiom is None:
            for key, axiom in self.axioms.properties.items():
                if (
                    key.lower() == pred_lower
                    or key.split(":")[-1].lower() == pred_lower.split(":")[-1]
                ):
                    prop_axiom = axiom
                    break

        if prop_axiom is None:
            return True, True, [], []

        domain_lower = [d.lower() for d in prop_axiom.domain]
        range_lower = [r.lower() for r in prop_axiom.range]

        domain_valid = not domain_lower or subj_lower in domain_lower
        range_valid = not range_lower or obj_lower in range_lower

        return domain_valid, range_valid, prop_axiom.domain, prop_axiom.range

    def check_property_chains(
        self,
        predicate: str,
        subject_id: str,
        subject_category: str,
        object_id: str,
        object_category: str,
        kg_edges: Sequence[Mapping[str, str]] | None = None,
    ) -> tuple[bool, list[dict[str, object]], bool, dict[str, object]]:
        """Check property chain axioms against KG context.

        Args:
            predicate: The claimed predicate
            subject_id: Subject entity ID
            subject_category: Subject category
            object_id: Object entity ID
            object_category: Object category
            kg_edges: Optional list of KG edges for chain checking

        Returns:
            Tuple of (chain_support, chain_paths, chain_conflict, conflict_details)
        """
        chain_support = False
        chain_paths: list[dict[str, object]] = []
        chain_conflict = False
        conflict_details: dict[str, object] = {}

        if not kg_edges:
            return chain_support, chain_paths, chain_conflict, conflict_details

        pred_lower = predicate.lower()
        subj_cat_lower = subject_category.lower()
        obj_cat_lower = object_category.lower()

        for chain_axiom in self.axioms.chains:
            inferred_pred_lower = chain_axiom.inferred_predicate.lower()
            if inferred_pred_lower not in pred_lower and pred_lower not in inferred_pred_lower:
                continue

            if chain_axiom.inferred_subject_type.lower() != subj_cat_lower:
                continue
            if chain_axiom.inferred_object_type.lower() != obj_cat_lower:
                continue

            if len(chain_axiom.chain) < 2:
                continue

            first_step = chain_axiom.chain[0]
            second_step = chain_axiom.chain[1]
            first_pred = first_step.get("predicate", "").lower()
            second_pred = second_step.get("predicate", "").lower()

            for edge in kg_edges:
                edge_subj = str(edge.get("subject", ""))
                edge_pred = str(edge.get("predicate", "")).lower()
                edge_obj = str(edge.get("object", ""))

                if edge_subj != subject_id:
                    continue
                if first_pred not in edge_pred and edge_pred not in first_pred:
                    continue

                mid_entity = edge_obj

                for edge2 in kg_edges:
                    edge2_subj = str(edge2.get("subject", ""))
                    edge2_pred = str(edge2.get("predicate", "")).lower()
                    edge2_obj = str(edge2.get("object", ""))

                    if edge2_subj != mid_entity:
                        continue
                    if second_pred not in edge2_pred and edge2_pred not in second_pred:
                        continue

                    if edge2_obj == object_id:
                        chain_support = True
                        chain_paths.append(
                            {
                                "chain_id": chain_axiom.id,
                                "mid_entity": mid_entity,
                                "via_predicates": [
                                    first_step.get("predicate"),
                                    second_step.get("predicate"),
                                ],
                            }
                        )
                    else:
                        chain_conflict = True
                        conflict_details = {
                            "chain_id": chain_axiom.id,
                            "expected_object": edge2_obj,
                            "claimed_object": object_id,
                            "mid_entity": mid_entity,
                        }

        return chain_support, chain_paths, chain_conflict, conflict_details

    def check_restrictions(
        self,
        predicate: str,
        subject_category: str,
        object_category: str,
        has_evidence: bool = False,
    ) -> tuple[bool, dict[str, object], bool, dict[str, object]]:
        """Check existential and universal restrictions.

        Returns:
            Tuple of (universal_violation, universal_details,
                     existential_violation, existential_details)
        """
        universal_violation = False
        universal_details: dict[str, object] = {}
        existential_violation = False
        existential_details: dict[str, object] = {}

        pred_lower = predicate.lower()
        subj_cat_lower = subject_category.lower()
        obj_cat_lower = object_category.lower()

        for restriction in self.axioms.restrictions:
            if restriction.kind == "universal":
                restr_prop = restriction.on_property.lower()
                if restr_prop not in pred_lower and pred_lower not in restr_prop:
                    continue

                restr_domain = restriction.domain.lower()
                if restr_domain and restr_domain != subj_cat_lower:
                    continue

                allowed = [f.lower() for f in restriction.allowed_fillers]
                if allowed and obj_cat_lower not in allowed:
                    universal_violation = True
                    universal_details = {
                        "restriction_id": restriction.id,
                        "property": restriction.on_property,
                        "expected_fillers": restriction.allowed_fillers,
                        "actual_filler": object_category,
                        "severity": restriction.severity,
                    }

            elif restriction.kind == "existential":
                if restriction.requires_evidence and not has_evidence:
                    restr_class = restriction.on_class.lower()
                    if restr_class == subj_cat_lower:
                        existential_violation = True
                        existential_details = {
                            "restriction_id": restriction.id,
                            "on_class": restriction.on_class,
                            "property": restriction.property,
                            "severity": restriction.severity,
                            "message": "Association should have supporting evidence",
                        }

        return universal_violation, universal_details, existential_violation, existential_details

    def check_all(
        self,
        predicate: str,
        subject_id: str,
        subject_category: str,
        object_id: str,
        object_category: str,
        kg_edges: Sequence[Mapping[str, str]] | None = None,
        has_evidence: bool = False,
    ) -> DLCheckResult:
        """Run all DL constraint checks.

        Returns:
            DLCheckResult with all constraint check results.
        """
        result = DLCheckResult()

        (
            result.disjoint_subject_violation,
            result.disjoint_object_violation,
            result.disjoint_pair_violation,
            result.disjoint_subject_types,
            result.disjoint_object_types,
            result.disjoint_pair_types,
        ) = self.check_disjointness(subject_category, object_category)

        (
            result.domain_valid,
            result.range_valid,
            result.domain_expected,
            result.range_expected,
        ) = self.check_domain_range(predicate, subject_category, object_category)

        (
            result.property_chain_support,
            result.property_chain_paths,
            result.property_chain_conflict,
            result.property_chain_conflict_details,
        ) = self.check_property_chains(
            predicate, subject_id, subject_category, object_id, object_category, kg_edges
        )

        (
            result.universal_violation,
            result.universal_violation_details,
            result.existential_violation,
            result.existential_violation_details,
        ) = self.check_restrictions(predicate, subject_category, object_category, has_evidence)

        return result


_default_checker: OntologyConstraintChecker | None = None


def get_default_checker() -> OntologyConstraintChecker:
    """Get or create the default constraint checker."""
    global _default_checker
    if _default_checker is None:
        _default_checker = OntologyConstraintChecker()
    return _default_checker


def check_dl_constraints(
    predicate: str,
    subject_id: str,
    subject_category: str,
    object_id: str,
    object_category: str,
    kg_edges: Sequence[Mapping[str, str]] | None = None,
    has_evidence: bool = False,
) -> dict[str, object]:
    """Convenience function to check all DL constraints and return facts dict.

    This is the main entry point for the pipeline to call.

    Returns:
        Dictionary of DL constraint facts for the rule engine.
    """
    checker = get_default_checker()
    result = checker.check_all(
        predicate=predicate,
        subject_id=subject_id,
        subject_category=subject_category,
        object_id=object_id,
        object_category=object_category,
        kg_edges=kg_edges,
        has_evidence=has_evidence,
    )
    return result.to_facts()
