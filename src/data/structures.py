"""
Data structures for the DiReCT Clinical Diagnostic Reasoning pipeline.

Ported from the official DiReCT repo (wbw520/DiReCT) utils.data_analysis,
adapted for clarity and type safety.

Structures:
    RecordNode   — A single node in the annotation entailment tree.
    InputContent — The 6 clinical text fields from a patient note.
    DiagnosticChain — The sequential reasoning chain extracted from the tree.
    KnowledgeGraph  — A disease-category knowledge graph (diagnostic tree + premises).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(Enum):
    """Type of node in the annotation entailment tree."""

    INPUT = "Input"  # Observation extracted verbatim from clinical text
    CAUSE = "Cause"  # Rationale linking an observation to a diagnosis step
    INTERMEDIA = "Intermedia"  # Intermediate diagnostic milestone or final PDD


# ---------------------------------------------------------------------------
# Core structures
# ---------------------------------------------------------------------------


@dataclass
class RecordNode:
    """A single node in the annotation entailment tree.

    Attributes:
        node_id: Unique identifier within this sample, e.g. "Input6", "Cause_1",
                 "Intermedia_2".
        content: The text content — clinical finding, rationale, or diagnosis name.
        node_type: One of Input / Cause / Intermedia.
        raw_key: The original key string from the JSON, e.g.
                 "hypoxia$Input1" or "Bacterial Pneumonia$Intermedia_4".
        parent_id: The node_id of the parent node (None for the root).
        child_ids: List of node_ids of child nodes.
        source_field: For Input nodes, which input field (1-6) the observation
                      was extracted from.  None for non-Input nodes.
        depth: Depth in the tree (root = 0).
    """

    node_id: str
    content: str
    node_type: NodeType
    raw_key: str
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    source_field: int | None = None
    depth: int = 0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_leaf(self) -> bool:
        return len(self.child_ids) == 0

    @property
    def is_observation(self) -> bool:
        return self.node_type == NodeType.INPUT

    @property
    def is_rationale(self) -> bool:
        return self.node_type == NodeType.CAUSE

    @property
    def is_diagnosis(self) -> bool:
        return self.node_type == NodeType.INTERMEDIA

    def __repr__(self) -> str:
        return (
            f"RecordNode(id={self.node_id!r}, type={self.node_type.value}, "
            f"content={self.content[:60]!r}{'...' if len(self.content) > 60 else ''})"
        )


@dataclass
class InputContent:
    """The 6 clinical text fields from a patient note.

    Fields:
        input1: Chief Complaint
        input2: History of Present Illness
        input3: Past Medical History
        input4: Family History
        input5: Physical Exam
        input6: Pertinent Results
    """

    input1: str = ""  # Chief Complaint
    input2: str = ""  # History of Present Illness
    input3: str = ""  # Past Medical History
    input4: str = ""  # Family History
    input5: str = ""  # Physical Exam
    input6: str = ""  # Pertinent Results

    FIELD_NAMES = {
        1: "Chief Complaint",
        2: "History of Present Illness",
        3: "Past Medical History",
        4: "Family History",
        5: "Physical Exam",
        6: "Pertinent Results",
    }

    def get_field(self, field_num: int) -> str:
        """Get clinical text for field number 1-6."""
        if field_num < 1 or field_num > 6:
            raise ValueError(f"Field number must be 1-6, got {field_num}")
        return getattr(self, f"input{field_num}")

    def get_all_text(self, separator: str = "\n\n") -> str:
        """Concatenate all non-empty fields into a single string."""
        parts = []
        for i in range(1, 7):
            text = self.get_field(i).strip()
            if text and text.lower() != "n/a":
                field_name = self.FIELD_NAMES[i]
                parts.append(f"[{field_name}]\n{text}")
        return separator.join(parts)

    def non_empty_fields(self) -> list[int]:
        """Return list of field numbers (1-6) that have non-trivial content."""
        result = []
        for i in range(1, 7):
            text = self.get_field(i).strip()
            if text and text.lower() != "n/a":
                result.append(i)
        return result

    def __repr__(self) -> str:
        filled = self.non_empty_fields()
        return f"InputContent(filled_fields={filled})"


@dataclass
class DiagnosticChain:
    """A single observation→rationale→diagnosis reasoning step.

    This is the flattened representation extracted by deduction_assemble():
        o → z → d
    where:
        o = clinical observation text
        z = rationale / reasoning
        r = source field identifier (which input1-6 field)
        d = associated diagnosis / intermediate conclusion
    """

    observation: str  # Clinical observation text (from Input node)
    rationale: str  # Reasoning text (from Cause node)
    source_field: int  # Which input field (1-6) the observation came from
    diagnosis: str  # The diagnosis this step supports (from Intermedia node)
    observation_node_id: str = ""  # Reference back to tree node
    rationale_node_id: str = ""
    diagnosis_node_id: str = ""

    def __repr__(self) -> str:
        return (
            f"DiagnosticChain(obs={self.observation[:40]!r}... → "
            f"diag={self.diagnosis[:40]!r}...)"
        )


@dataclass
class KGNode:
    """A single node in a disease-category knowledge graph's diagnostic tree."""

    name: str
    children: list[KGNode] = field(default_factory=list)
    premise: str | None = None  # The textbook knowledge/criteria for this node

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_all_leaves(self) -> list[KGNode]:
        """Recursively collect all leaf (PDD) nodes."""
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves

    def __repr__(self) -> str:
        return f"KGNode({self.name!r}, children={len(self.children)})"


@dataclass
class KnowledgeGraph:
    """A disease-category knowledge graph.

    Contains:
        category: Disease category name (e.g. "Pneumonia").
        diagnostic_tree: The root KGNode of the diagnostic flowchart
                         (from "diagnostic" key in JSON).
        knowledge: Dict mapping node name → textbook premises
                   (from "knowledge" key in JSON).
    """

    category: str
    diagnostic_tree: KGNode
    knowledge: dict[str, Any]

    @property
    def root_name(self) -> str:
        return self.diagnostic_tree.name

    def get_premise(self, node_name: str) -> str | None:
        """Get the textbook knowledge/premise for a given node name."""
        val = self.knowledge.get(node_name)
        if val is None:
            return None
        if isinstance(val, dict):
            # Root-level nodes have structured premises (Risk Factors, Symptoms, Signs)
            parts = []
            for k, v in val.items():
                parts.append(f"{k}: {v}")
            return "\n".join(parts)
        return str(val)

    def get_all_pdd_names(self) -> list[str]:
        """Get all possible Primary Discharge Diagnosis names (leaf nodes)."""
        return [node.name for node in self.diagnostic_tree.get_all_leaves()]

    def get_all_node_names(self) -> list[str]:
        """Get all node names in the diagnostic tree (BFS order)."""
        names = []
        queue = [self.diagnostic_tree]
        while queue:
            node = queue.pop(0)
            names.append(node.name)
            queue.extend(node.children)
        return names

    def __repr__(self) -> str:
        pdds = self.get_all_pdd_names()
        return (
            f"KnowledgeGraph(category={self.category!r}, "
            f"root={self.root_name!r}, pdds={pdds})"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Regex to parse annotation keys like "content text$NodeType_id"
# Examples:
#   "hypoxia$Input1"  → content="hypoxia", type="Input", id="1"
#   "Bacterial Pneumonia$Intermedia_4" → content="Bacterial Pneumonia", type="Intermedia", id="4"
#   "Some rationale$Cause_1" → content="Some rationale", type="Cause", id="1"
_KEY_PATTERN = re.compile(
    r"^([\s\S]+?)\$(Input|Cause|Intermedia)_?(\d+)\Z"
)


def parse_annotation_key(key: str) -> tuple[str, NodeType, str]:
    """Parse an annotation key into (content, node_type, node_id).

    The key format is: "<content>$<NodeType>_<id>" or "<content>$<NodeType><id>"

    For Input nodes, the id is the input field number (1-6).
    For Cause/Intermedia nodes, the id is a sequential counter.

    Returns:
        (content, node_type, unique_node_id)

    Raises:
        ValueError: If the key doesn't match the expected pattern.
    """
    match = _KEY_PATTERN.match(key)
    if not match:
        raise ValueError(f"Cannot parse annotation key: {key!r}")

    content = match.group(1).strip()
    type_str = match.group(2)
    id_str = match.group(3) or ""

    node_type = NodeType(type_str)

    # Build a unique node_id
    if node_type == NodeType.INPUT:
        # For Input nodes, id is the field number (e.g., "Input6" → source_field=6)
        # The node_id includes content hash for uniqueness (multiple observations
        # can come from the same field)
        node_id = f"{type_str}{id_str}"
    else:
        node_id = f"{type_str}_{id_str}" if id_str else type_str

    return content, node_type, node_id
