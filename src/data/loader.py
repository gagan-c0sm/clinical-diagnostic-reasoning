"""
Data loader for DiReCT (Diagnostic Reasoning for Clinical Notes) dataset.

Ports and extends the official repo's utils.data_analysis (cal_a_json,
deduction_assemble) to produce structured Python objects instead of raw dicts.

Usage:
    from src.data import DiReCTDataset

    dataset = DiReCTDataset("mimic-iv-ext-direct-1.0.0")
    print(dataset.summary())

    for sample in dataset:
        print(sample.note_id, sample.disease_category, sample.pdd)
        print(sample.inputs)
        print(sample.record_nodes)
        print(sample.diagnostic_chains)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .structures import (
    DiagnosticChain,
    InputContent,
    KGNode,
    KnowledgeGraph,
    NodeType,
    RecordNode,
    parse_annotation_key,
)
from .labels import normalize_diagnosis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_FIELDS = [f"input{i}" for i in range(1, 7)]


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClinicalSample:
    """A single annotated clinical note from the DiReCT dataset.

    Holds the parsed annotation tree, input fields, and metadata.
    """

    note_id: str  # e.g. "13691292-DS-3"
    disease_category: str  # e.g. "Pneumonia"
    pdd: str  # Primary Discharge Diagnosis, e.g. "Bacterial Pneumonia"
    file_path: Path

    inputs: InputContent = field(default_factory=InputContent)
    record_nodes: dict[str, RecordNode] = field(default_factory=dict)
    root_node_id: str | None = None

    @property
    def sample_id(self) -> str:
        """Annotation identity; a note can have multiple annotated diagnoses."""
        parts = self.file_path.parts
        if "Finished" in parts:
            return "/".join(parts[parts.index("Finished") + 1:])
        return f"{self.disease_category}/{self.pdd}/{self.file_path.name}"

    @property
    def patient_id(self) -> str:
        return self.note_id.split("-", 1)[0]

    @property
    def observations(self) -> list[RecordNode]:
        """All Input (observation) nodes."""
        return [n for n in self.record_nodes.values() if n.is_observation]

    @property
    def rationales(self) -> list[RecordNode]:
        """All Cause (rationale) nodes."""
        return [n for n in self.record_nodes.values() if n.is_rationale]

    @property
    def diagnoses(self) -> list[RecordNode]:
        """All Intermedia (diagnosis) nodes."""
        return [n for n in self.record_nodes.values() if n.is_diagnosis]

    @property
    def root_node(self) -> RecordNode | None:
        """The root node of the entailment tree."""
        if self.root_node_id and self.root_node_id in self.record_nodes:
            return self.record_nodes[self.root_node_id]
        return None

    @property
    def final_diagnosis_node(self) -> RecordNode | None:
        """The root of the tree is typically the final (most specific) diagnosis."""
        return self.root_node

    def get_diagnostic_chains(self) -> list[DiagnosticChain]:
        """Extract flattened observation→rationale→diagnosis chains.

        This is equivalent to the paper's deduction_assemble() function.
        Walks the tree to find all (Input → Cause → Intermedia) triples.
        """
        chains = []
        self._extract_chains(self.root_node_id, chains)
        return chains

    def _extract_chains(
        self, node_id: str | None, chains: list[DiagnosticChain]
    ) -> None:
        """Recursively extract reasoning chains from the tree."""
        if node_id is None or node_id not in self.record_nodes:
            return

        node = self.record_nodes[node_id]

        if node.is_rationale:
            # A Cause node bridges Input(s) below it to Intermedia above it
            # Find the parent Intermedia
            parent = self.record_nodes.get(node.parent_id) if node.parent_id else None
            diagnosis_text = parent.content if parent and parent.is_diagnosis else ""
            diagnosis_id = parent.node_id if parent else ""

            # Find child Input nodes
            for child_id in node.child_ids:
                child = self.record_nodes.get(child_id)
                if child and child.is_observation:
                    chains.append(
                        DiagnosticChain(
                            observation=child.content,
                            rationale=node.content,
                            source_field=child.source_field or 0,
                            diagnosis=diagnosis_text,
                            observation_node_id=child.node_id,
                            rationale_node_id=node.node_id,
                            diagnosis_node_id=diagnosis_id,
                        )
                    )

        # Recurse into children
        for child_id in node.child_ids:
            self._extract_chains(child_id, chains)

    def tree_depth(self) -> int:
        """Maximum depth of the entailment tree."""
        if not self.record_nodes:
            return 0
        return max(n.depth for n in self.record_nodes.values())

    def __repr__(self) -> str:
        return (
            f"ClinicalSample(id={self.note_id!r}, "
            f"category={self.disease_category!r}, "
            f"pdd={self.pdd!r}, "
            f"nodes={len(self.record_nodes)}, "
            f"observations={len(self.observations)}, "
            f"depth={self.tree_depth()})"
        )


# ---------------------------------------------------------------------------
# JSON parsing: cal_a_json equivalent
# ---------------------------------------------------------------------------


def _parse_tree_recursive(
    tree_dict: dict[str, Any],
    parent_id: str | None,
    depth: int,
    nodes: dict[str, RecordNode],
    _counter: dict[str, int],
) -> str | None:
    """Recursively parse the nested annotation JSON into flat RecordNode dicts.

    The DiReCT annotation JSON has keys like:
        "Bacterial Pneumonia$Intermedia_4": {
            "Rationale text$Cause_1": {
                "observation text$Input6": {}
            }
        }

    Each key encodes content + node type. Nesting encodes parent→child relationships.
    The tree is inverted: the root (final diagnosis) is the outermost key,
    and leaf nodes (observations) are the innermost.

    Args:
        tree_dict: The current level of the nested dict.
        parent_id: node_id of the parent (None at root level).
        depth: Current tree depth.
        nodes: Accumulator dict of node_id → RecordNode.
        _counter: Mutable counter dict for generating unique IDs when needed.

    Returns:
        The node_id of the node parsed at this level, or None.
    """
    root_id = None

    for raw_key, children_dict in tree_dict.items():
        # Skip input fields (input1-input6) that sit at the same level
        if raw_key in INPUT_FIELDS:
            continue

        try:
            content, node_type, base_id = parse_annotation_key(raw_key)
        except ValueError:
            logger.warning(f"Skipping unparseable key: {raw_key!r}")
            continue

        # Generate a unique node_id (same base_id can appear at different levels)
        # Use a counter to disambiguate
        counter_key = f"{node_type.value}_{base_id}"
        if counter_key in _counter:
            _counter[counter_key] += 1
            unique_id = f"{base_id}_{_counter[counter_key]}"
        else:
            _counter[counter_key] = 0
            unique_id = base_id

        # Determine source_field for Input nodes
        source_field = None
        if node_type == NodeType.INPUT:
            # Extract the field number from the id (e.g., "Input6" → 6)
            match = re.search(r"(\d+)", base_id)
            if match:
                source_field = int(match.group(1))

        node = RecordNode(
            node_id=unique_id,
            content=content,
            node_type=node_type,
            raw_key=raw_key,
            parent_id=parent_id,
            child_ids=[],
            source_field=source_field,
            depth=depth,
        )

        nodes[unique_id] = node

        # Wire up parent → child
        if parent_id is not None and parent_id in nodes:
            nodes[parent_id].child_ids.append(unique_id)

        # The first node at depth 0 is the root
        if root_id is None and depth == 0:
            root_id = unique_id
        elif root_id is None:
            root_id = unique_id

        # Recurse into children
        if isinstance(children_dict, dict) and children_dict:
            _parse_tree_recursive(children_dict, unique_id, depth + 1, nodes, _counter)

    return root_id


def load_sample(file_path: Path | str, disease_category: str, pdd: str) -> ClinicalSample:
    """Load and parse a single annotated clinical note JSON.

    This is the equivalent of the official repo's cal_a_json().

    Args:
        file_path: Path to the note JSON file.
        disease_category: Disease category name (from folder structure).
        pdd: Primary Discharge Diagnosis (from subfolder name).

    Returns:
        A fully parsed ClinicalSample.
    """
    file_path = Path(file_path)
    note_id = file_path.stem  # e.g. "13691292-DS-3"

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract input fields
    inputs = InputContent(
        input1=raw_data.get("input1", ""),
        input2=raw_data.get("input2", ""),
        input3=raw_data.get("input3", ""),
        input4=raw_data.get("input4", ""),
        input5=raw_data.get("input5", ""),
        input6=raw_data.get("input6", ""),
    )

    # Parse the annotation tree (everything except input1-6 keys)
    tree_dict = {k: v for k, v in raw_data.items() if k not in INPUT_FIELDS}

    nodes: dict[str, RecordNode] = {}
    counter: dict[str, int] = {}
    root_id = _parse_tree_recursive(tree_dict, None, 0, nodes, counter)

    sample = ClinicalSample(
        note_id=note_id,
        disease_category=disease_category,
        pdd=pdd,
        file_path=file_path,
        inputs=inputs,
        record_nodes=nodes,
        root_node_id=root_id,
    )

    return sample


# ---------------------------------------------------------------------------
# Knowledge graph loading
# ---------------------------------------------------------------------------


def _parse_kg_tree(tree_dict: dict) -> KGNode:
    """Recursively parse a diagnostic tree dict into KGNode objects."""
    # The dict has exactly one root key
    if not tree_dict:
        raise ValueError("Empty diagnostic tree")

    # Handle both dict and list values
    for name, children in tree_dict.items():
        child_nodes = []
        if isinstance(children, dict):
            for child_name, grandchildren in children.items():
                if isinstance(grandchildren, dict) and grandchildren:
                    child_nodes.append(_parse_kg_tree({child_name: grandchildren}))
                else:
                    # Leaf node (empty dict or list)
                    child_nodes.append(KGNode(name=child_name))
        elif isinstance(children, list):
            # Leaf node
            pass

        return KGNode(name=name, children=child_nodes)


def load_diagnostic_kg(file_path: Path | str) -> KnowledgeGraph:
    """Load a disease-category knowledge graph JSON.

    Args:
        file_path: Path to the KG JSON file (e.g., "Pneumonia.json").

    Returns:
        A KnowledgeGraph with parsed diagnostic tree and premises.
    """
    file_path = Path(file_path)
    category = file_path.stem  # e.g. "Pneumonia"

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Parse the diagnostic flowchart tree
    diagnostic_dict = raw_data.get("diagnostic", {})
    tree = _parse_kg_tree(diagnostic_dict)

    # Extract knowledge premises
    knowledge = raw_data.get("knowledge", {})

    # Attach premises to tree nodes
    _attach_premises(tree, knowledge)

    kg = KnowledgeGraph(
        category=category,
        diagnostic_tree=tree,
        knowledge=knowledge,
    )

    return kg


def _attach_premises(node: KGNode, knowledge: dict) -> None:
    """Recursively attach premise text from knowledge dict to tree nodes."""
    if node.name in knowledge:
        val = knowledge[node.name]
        if isinstance(val, str):
            node.premise = val
        elif isinstance(val, dict):
            # Structured premise (Risk Factors, Symptoms, Signs)
            parts = [f"{k}: {v}" for k, v in val.items()]
            node.premise = "\n".join(parts)
    for child in node.children:
        _attach_premises(child, knowledge)


# ---------------------------------------------------------------------------
# Bulk loading
# ---------------------------------------------------------------------------


def load_all_samples(
    samples_dir: Path | str,
) -> list[ClinicalSample]:
    """Load all annotated clinical note samples from the dataset.

    Traverses the folder structure:
        samples/Finished/<Disease Category>/<PDD subfolder>/note_id.json

    Some categories have PDD subfolders, some don't (notes directly under category).

    Args:
        samples_dir: Path to the "samples/Finished" directory.

    Returns:
        List of all parsed ClinicalSamples, sorted by disease category then note_id.
    """
    samples_dir = Path(samples_dir)
    samples = []

    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    for category_dir in sorted(samples_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        disease_category = category_dir.name

        # Check if this category has PDD subfolders or notes directly
        json_files = list(category_dir.glob("*.json"))
        sub_dirs = [d for d in category_dir.iterdir() if d.is_dir()]

        if json_files:
            # Notes directly under category (no PDD subfolder)
            for json_file in sorted(json_files):
                pdd = disease_category  # PDD same as category
                try:
                    sample = load_sample(json_file, disease_category, pdd)
                    samples.append(sample)
                except Exception as e:
                    logger.error(
                        f"Failed to load {json_file}: {e}", exc_info=True
                    )

        for sub_dir in sorted(sub_dirs):
            pdd = sub_dir.name
            for json_file in sorted(sub_dir.glob("*.json")):
                try:
                    sample = load_sample(json_file, disease_category, pdd)
                    samples.append(sample)
                except Exception as e:
                    logger.error(
                        f"Failed to load {json_file}: {e}", exc_info=True
                    )

    logger.info(f"Loaded {len(samples)} clinical samples from {samples_dir}")
    return samples


def load_all_kgs(kg_dir: Path | str) -> dict[str, KnowledgeGraph]:
    """Load all disease-category knowledge graphs.

    Args:
        kg_dir: Path to the "diagnostic_kg/Diagnosis_flowchart" directory.

    Returns:
        Dict mapping disease category name → KnowledgeGraph.
    """
    kg_dir = Path(kg_dir)
    kgs = {}

    if not kg_dir.exists():
        raise FileNotFoundError(f"Knowledge graph directory not found: {kg_dir}")

    for json_file in sorted(kg_dir.glob("*.json")):
        try:
            kg = load_diagnostic_kg(json_file)
            kgs[kg.category] = kg
        except Exception as e:
            logger.error(f"Failed to load KG {json_file}: {e}", exc_info=True)

    logger.info(f"Loaded {len(kgs)} knowledge graphs from {kg_dir}")
    return kgs


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


class DiReCTDataset:
    """Full DiReCT dataset: samples + knowledge graphs.

    Usage:
        dataset = DiReCTDataset("mimic-iv-ext-direct-1.0.0")
        print(dataset.summary())

        for sample in dataset:
            kg = dataset.get_kg(sample.disease_category)
            chains = sample.get_diagnostic_chains()
    """

    def __init__(self, root_dir: Path | str):
        """Load the complete DiReCT dataset.

        Args:
            root_dir: Path to the "mimic-iv-ext-direct-1.0.0" directory.
        """
        self.root_dir = Path(root_dir)
        self.samples_dir = self.root_dir / "samples" / "Finished"
        self.kg_dir = self.root_dir / "diagnostic_kg" / "Diagnosis_flowchart"

        # Load everything
        self.knowledge_graphs = load_all_kgs(self.kg_dir)
        self.samples = load_all_samples(self.samples_dir)

        # Index samples by various keys
        self._by_id: dict[str, ClinicalSample] = {}
        self._by_note: dict[str, list[ClinicalSample]] = {}
        self._by_category: dict[str, list[ClinicalSample]] = {}
        self._by_pdd: dict[str, list[ClinicalSample]] = {}

        for sample in self.samples:
            self._by_id[sample.sample_id] = sample
            self._by_note.setdefault(sample.note_id, []).append(sample)

            if sample.disease_category not in self._by_category:
                self._by_category[sample.disease_category] = []
            self._by_category[sample.disease_category].append(sample)

            if sample.pdd not in self._by_pdd:
                self._by_pdd[sample.pdd] = []
            self._by_pdd[sample.pdd].append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[ClinicalSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> ClinicalSample:
        return self.samples[idx]

    def get_sample(self, sample_id: str) -> ClinicalSample | None:
        """Look up a unique annotation, rejecting ambiguous legacy note IDs."""
        if sample_id in self._by_id:
            return self._by_id[sample_id]
        matches = self._by_note.get(sample_id, [])
        if len(matches) > 1:
            raise ValueError(f"Ambiguous note ID {sample_id!r}; use sample_id")
        return matches[0] if matches else None

    def get_kg(self, disease_category: str) -> KnowledgeGraph | None:
        """Get the knowledge graph for a disease category."""
        return self.knowledge_graphs.get(disease_category)

    def get_samples_by_category(self, category: str) -> list[ClinicalSample]:
        """Get all samples for a given disease category."""
        return self._by_category.get(category, [])

    def get_samples_by_pdd(self, pdd: str) -> list[ClinicalSample]:
        """Get all samples for a given Primary Discharge Diagnosis."""
        return self._by_pdd.get(pdd, [])

    @property
    def categories(self) -> list[str]:
        """All disease categories (sorted)."""
        return sorted(self._by_category.keys())

    @property
    def pdds(self) -> list[str]:
        """All Primary Discharge Diagnoses (sorted)."""
        return sorted(self._by_pdd.keys())

    def category_distribution(self) -> dict[str, int]:
        """Number of samples per disease category."""
        return {cat: len(samples) for cat, samples in sorted(self._by_category.items())}

    def pdd_distribution(self) -> dict[str, int]:
        """Number of samples per PDD."""
        return {pdd: len(samples) for pdd, samples in sorted(self._by_pdd.items())}

    def summary(self) -> str:
        """Human-readable dataset summary."""
        lines = [
            "=" * 70,
            "DiReCT Dataset Summary",
            "=" * 70,
            f"Root directory:        {self.root_dir}",
            f"Total samples:         {len(self.samples)}",
            f"Disease categories:    {len(self._by_category)}",
            f"Distinct PDDs:         {len(self._by_pdd)}",
            f"Knowledge graphs:      {len(self.knowledge_graphs)}",
            "",
            "Samples per category:",
        ]

        for cat, count in sorted(
            self.category_distribution().items(), key=lambda x: -x[1]
        ):
            kg = self.knowledge_graphs.get(cat)
            pdd_count = len(kg.get_all_pdd_names()) if kg else "?"
            lines.append(f"  {cat:<40s} {count:>4d} samples, {pdd_count} PDDs")

        # Aggregate stats on tree structure
        total_obs = sum(len(s.observations) for s in self.samples)
        total_rationales = sum(len(s.rationales) for s in self.samples)
        total_diagnoses = sum(len(s.diagnoses) for s in self.samples)
        avg_depth = (
            sum(s.tree_depth() for s in self.samples) / len(self.samples)
            if self.samples
            else 0
        )

        lines.extend([
            "",
            "Aggregate annotation statistics:",
            f"  Total observation nodes:  {total_obs}",
            f"  Total rationale nodes:    {total_rationales}",
            f"  Total diagnosis nodes:    {total_diagnoses}",
            f"  Average tree depth:       {avg_depth:.1f}",
            f"  Avg observations/sample:  {total_obs / len(self.samples):.1f}"
            if self.samples
            else "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def validate(self) -> list[str]:
        """Run validation checks on the loaded dataset.

        Returns a list of warning/error messages. Empty list = all good.
        """
        issues = []

        # Check that every sample's disease category has a matching KG
        for sample in self.samples:
            if sample.disease_category not in self.knowledge_graphs:
                issues.append(
                    f"Sample {sample.note_id}: disease category "
                    f"{sample.disease_category!r} has no matching KG"
                )

        # Check that every sample's PDD appears in its category's KG
        for sample in self.samples:
            kg = self.knowledge_graphs.get(sample.disease_category)
            if kg:
                valid_pdds = kg.get_all_node_names()
                if normalize_diagnosis(sample.pdd) not in {normalize_diagnosis(n) for n in valid_pdds} and sample.pdd != sample.disease_category:
                    issues.append(
                        f"Sample {sample.note_id}: PDD {sample.pdd!r} not found "
                        f"in KG for {sample.disease_category!r}. "
                        f"Valid PDDs: {valid_pdds}"
                    )

        # Check tree connectivity
        for sample in self.samples:
            if sample.root_node_id is None:
                issues.append(
                    f"Sample {sample.note_id}: no root node found"
                )
            elif sample.root_node_id not in sample.record_nodes:
                issues.append(
                    f"Sample {sample.note_id}: root node ID "
                    f"{sample.root_node_id!r} not in record_nodes"
                )

            # Check for orphan nodes
            if sample.root_node_id:
                reachable = set()
                _collect_reachable(
                    sample.root_node_id, sample.record_nodes, reachable
                )
                orphans = set(sample.record_nodes.keys()) - reachable
                if orphans:
                    issues.append(
                        f"Sample {sample.note_id}: {len(orphans)} orphan node(s) "
                        f"not reachable from root: {orphans}"
                    )

            # Check that observations have source_field set
            for node in sample.observations:
                if node.source_field is None:
                    issues.append(
                        f"Sample {sample.note_id}, node {node.node_id}: "
                        f"Input node has no source_field"
                    )

        return issues


def _collect_reachable(
    node_id: str, nodes: dict[str, RecordNode], visited: set[str]
) -> None:
    """BFS/DFS to collect all reachable node IDs from a starting node."""
    if node_id in visited or node_id not in nodes:
        return
    visited.add(node_id)
    for child_id in nodes[node_id].child_ids:
        _collect_reachable(child_id, nodes, visited)
