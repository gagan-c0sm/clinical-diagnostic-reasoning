"""
Dataset downloader and preprocessor for supplementary training datasets.

Downloads and formats:
  - MedCaseReasoning (Stanford/zou-lab): 14,489 diagnostic reasoning cases
  - Medical-O1-Reasoning-SFT (FreedomIntelligence): Chain-of-thought medical reasoning

All downloads are cached inside the project's data/ directory.

Usage:
    python -m scripts.download_datasets --output-dir data/supplementary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MedCaseReasoning (zou-lab/MedCaseReasoning)
# ---------------------------------------------------------------------------

MEDCASE_HF_REPO = "zou-lab/MedCaseReasoning"
MEDCASE_REVISION = "469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
MEDCASE_DESCRIPTION = (
    "MedCaseReasoning: 14,489 diagnostic cases from Stanford. "
    "Each case has a detailed patient presentation, numbered clinician "
    "reasoning steps, and a final diagnosis. Covers 30+ specialties."
)


def download_medcase_reasoning(
    output_dir: Path,
    cache_dir: Path,
    max_samples: int | None = None,
) -> dict[str, int]:
    """Download and preprocess MedCaseReasoning dataset.

    Args:
        output_dir: Where to save processed JSONL files.
        cache_dir: HuggingFace cache directory (inside project).
        max_samples: Optional limit on number of samples to save.

    Returns:
        Dict with split names and sample counts.
    """
    from datasets import load_dataset

    logger.info(f"Downloading MedCaseReasoning from {MEDCASE_HF_REPO}...")
    logger.info(f"  Cache: {cache_dir}")

    ds = load_dataset(MEDCASE_HF_REPO, "default", revision=MEDCASE_REVISION,
                      cache_dir=str(cache_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for split_name in ds:
        split = ds[split_name]
        out_path = output_dir / f"medcase_reasoning_{split_name}.jsonl"

        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for i, example in enumerate(split):
                if max_samples and i >= max_samples:
                    break

                # Format into instruction-response pairs for SFT
                record = format_medcase_for_sft(example)
                if record:
                    record.update(source=MEDCASE_HF_REPO, source_revision=MEDCASE_REVISION,
                                  source_split=split_name, source_row=i, task_type="medical_reasoning")
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n += 1

        if not n:
            raise ValueError(f"No usable MedCaseReasoning records in {split_name}")
        counts[split_name] = n
        logger.info(f"  Saved {n} samples to {out_path}")

    # Write a metadata file
    meta = {
        "source": MEDCASE_HF_REPO,
        "revision": MEDCASE_REVISION,
        "description": MEDCASE_DESCRIPTION,
        "splits": counts,
        "format": "jsonl",
        "fields": [
            "instruction", "input", "output",
            "case_prompt", "diagnostic_reasoning", "final_diagnosis",
            "pmcid", "specialties",
        ],
    }
    meta_path = output_dir / "medcase_reasoning_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return counts


def format_medcase_for_sft(example: dict[str, Any]) -> dict[str, Any] | None:
    """Format a MedCaseReasoning example into an SFT training record.

    Transforms:
        case_prompt → instruction (the patient case)
        diagnostic_reasoning → reasoning steps in the output
        final_diagnosis → final answer in the output

    Returns None if the example is malformed.
    """
    case_prompt = (example.get("case_prompt") or "").strip()
    reasoning = (example.get("diagnostic_reasoning") or "").strip()
    diagnosis = (example.get("final_diagnosis") or "").strip()

    if not case_prompt or not diagnosis:
        return None

    instruction = (
        "You are a clinical diagnostic reasoning assistant. Analyze the "
        "following patient case and provide step-by-step diagnostic reasoning "
        "leading to the final diagnosis."
    )

    output_parts = []
    if reasoning:
        output_parts.append(f"## Diagnostic Reasoning\n{reasoning}")
    output_parts.append(f"## Final Diagnosis\n{diagnosis}")
    output_text = "\n\n".join(output_parts)

    record = {
        "instruction": instruction,
        "input": case_prompt,
        "output": output_text,
        # Preserve original fields for reference
        "case_prompt": case_prompt,
        "diagnostic_reasoning": reasoning,
        "final_diagnosis": diagnosis,
    }

    # Copy metadata fields if present
    for key in ["pmcid", "title", "journal", "article_link", "publication_date"]:
        if key in example and example[key]:
            record[key] = example[key]

    return record


# ---------------------------------------------------------------------------
# Medical-O1-Reasoning-SFT (FreedomIntelligence)
# ---------------------------------------------------------------------------

MEDICAL_O1_HF_REPO = "FreedomIntelligence/medical-o1-reasoning-SFT"
MEDICAL_O1_REVISION = "fc2c9e8a37b38f38da6d449564a8c350b244aef4"
MEDICAL_O1_DESCRIPTION = (
    "Medical-O1-Reasoning-SFT: Chain-of-thought medical reasoning dataset. "
    "Contains medical questions paired with explicit multi-step reasoning "
    "chains and verified answers. Used to train HuatuoGPT-o1."
)


def download_medical_o1(
    output_dir: Path,
    cache_dir: Path,
    max_samples: int | None = None,
    language_filter: str = "en",
) -> dict[str, int]:
    """Download and preprocess Medical-O1-Reasoning-SFT dataset.

    Args:
        output_dir: Where to save processed JSONL files.
        cache_dir: HuggingFace cache directory (inside project).
        max_samples: Optional limit on number of samples to save.
        language_filter: Language to filter for ("en", "zh", or None for all).

    Returns:
        Dict with split names and sample counts.
    """
    from datasets import load_dataset

    logger.info(f"Downloading Medical-O1-Reasoning-SFT from {MEDICAL_O1_HF_REPO}...")
    logger.info(f"  Cache: {cache_dir}")

    if language_filter not in ("en", "zh"):
        raise ValueError("Select the official en or zh subset explicitly")
    ds = load_dataset(MEDICAL_O1_HF_REPO, language_filter,
                      revision=MEDICAL_O1_REVISION, cache_dir=str(cache_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for split_name in ds:
        split = ds[split_name]
        out_path = output_dir / f"medical_o1_{split_name}.jsonl"

        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for i, example in enumerate(split):
                if max_samples and i >= max_samples:
                    break

                record = format_medical_o1_for_sft(example, language_filter)
                if record:
                    record.update(source=MEDICAL_O1_HF_REPO, source_revision=MEDICAL_O1_REVISION,
                                  source_split=split_name, source_row=i, task_type="medical_reasoning")
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n += 1

        if not n:
            raise ValueError(f"No usable Medical-O1 records in {split_name}")
        counts[split_name] = n
        logger.info(f"  Saved {n} samples to {out_path}")

    # Write metadata
    meta = {
        "source": MEDICAL_O1_HF_REPO,
        "revision": MEDICAL_O1_REVISION,
        "description": MEDICAL_O1_DESCRIPTION,
        "splits": counts,
        "format": "jsonl",
        "language_filter": language_filter,
    }
    meta_path = output_dir / "medical_o1_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return counts


def format_medical_o1_for_sft(
    example: dict[str, Any],
    language_filter: str | None = "en",
) -> dict[str, Any] | None:
    """Format a Medical-O1 example into an SFT training record.

    The dataset uses a "conversations" format with role-based messages.
    We extract the user question and assistant response with CoT.

    Returns None if the example should be skipped (wrong language, malformed).
    """
    # The dataset may have different formats; handle both
    conversations = example.get("conversations", [])

    if conversations:
        # Conversation format: list of {role, content} or {from, value}
        user_text = ""
        assistant_text = ""

        for msg in conversations:
            role = msg.get("role") or msg.get("from", "")
            content = msg.get("content") or msg.get("value", "")

            if role in ("user", "human"):
                user_text = content.strip()
            elif role in ("assistant", "gpt"):
                assistant_text = content.strip()

        if not user_text or not assistant_text:
            return None

        # Simple language detection: check for predominantly ASCII for English
        if language_filter == "en":
            ascii_ratio = sum(1 for c in user_text if ord(c) < 128) / max(
                len(user_text), 1
            )
            if ascii_ratio < 0.7:
                return None

        return {
            "instruction": (
                "You are a medical reasoning assistant. Analyze the following "
                "clinical question step-by-step and provide a thorough answer."
            ),
            "input": user_text,
            "output": assistant_text,
        }

    # Alternative format: direct question/response fields
    question = (example.get("Question") or example.get("question") or "").strip()
    response = (example.get("Response") or example.get("response") or "").strip()
    cot = example.get("Complex_CoT") or example.get("complex_cot") or example.get("chain_of_thought") or ""

    if not question:
        return None

    if language_filter == "en":
        ascii_ratio = sum(1 for c in question if ord(c) < 128) / max(
            len(question), 1
        )
        if ascii_ratio < 0.7:
            return None

    output_parts = []
    if cot:
        output_parts.append(cot.strip())
    if response:
        output_parts.append(response.strip())

    if not output_parts:
        return None

    return {
        "instruction": (
            "You are a medical reasoning assistant. Analyze the following "
            "clinical question step-by-step and provide a thorough answer."
        ),
        "input": question,
        "output": "\n\n".join(output_parts),
    }


# ---------------------------------------------------------------------------
# DiReCT-format training data generator
# ---------------------------------------------------------------------------


def generate_direct_sft_data(
    samples: list,  # list[ClinicalSample]
    knowledge_graphs: dict,  # dict[str, KnowledgeGraph]
    output_path: Path,
) -> int:
    """Convert DiReCT annotated samples into SFT training records.

    Each record teaches the model to generate an entailment tree given
    clinical inputs and a disease knowledge graph.

    Args:
        samples: List of ClinicalSample objects (training split only).
        knowledge_graphs: Dict of category → KnowledgeGraph.
        output_path: Path to write the JSONL file.

    Returns:
        Number of records written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            kg = knowledge_graphs.get(sample.disease_category)
            if kg is None:
                logger.warning(
                    f"No KG for {sample.disease_category}, skipping {sample.note_id}"
                )
                continue

            record = _format_direct_sample_for_sft(sample, kg)
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1

    logger.info(f"Generated {n} DiReCT-format SFT records → {output_path}")
    return n


def _format_direct_sample_for_sft(sample, kg) -> dict[str, Any] | None:
    """Format a single DiReCT sample into an SFT instruction/input/output record."""

    # Build the clinical note input
    clinical_note = sample.inputs.get_all_text()
    if not clinical_note.strip():
        return None

    # Build the KG context
    kg_context = _format_kg_context(kg)

    # Build the target output: the entailment tree as structured JSON
    tree_json = _build_entailment_tree_json(sample)
    if not tree_json:
        return None

    instruction = (
        "You are a clinical diagnostic reasoning system. Given the clinical "
        "note below and the disease knowledge graph, construct a diagnostic "
        "entailment tree. The tree should trace from clinical observations "
        "through causal rationales to intermediate and final diagnoses.\n\n"
        "Output a JSON object where each key is a node in the tree. Each node "
        "has: content, type (Input/Cause/Intermedia), parent, children, and "
        "for Input nodes a source_field (1-6)."
    )

    input_text = (
        f"## Clinical Note\n{clinical_note}\n\n"
        f"## Disease Category\n{sample.disease_category}\n\n"
        f"## Diagnostic Knowledge Graph\n{kg_context}"
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": json.dumps(tree_json, ensure_ascii=False, indent=2),
        # Metadata for analysis
        "note_id": sample.note_id,
        "sample_id": sample.sample_id,
        "patient_id": sample.patient_id,
        "disease_category": sample.disease_category,
        "pdd": sample.pdd,
    }


def _format_kg_context(kg) -> str:
    """Format a knowledge graph into a text context for the prompt."""
    lines = []

    # Diagnostic tree structure
    lines.append("### Diagnostic Tree")
    _format_kg_tree_text(kg.diagnostic_tree, lines, indent=0)

    # Knowledge premises
    lines.append("\n### Diagnostic Criteria")
    for node_name in kg.get_all_node_names():
        premise = kg.get_premise(node_name)
        if premise:
            lines.append(f"- **{node_name}**: {premise}")

    return "\n".join(lines)


def _format_kg_tree_text(node, lines: list[str], indent: int = 0) -> None:
    """Recursively format the KG tree into indented text."""
    prefix = "  " * indent
    suffix = " (leaf/PDD)" if node.is_leaf else ""
    lines.append(f"{prefix}- {node.name}{suffix}")
    for child in node.children:
        _format_kg_tree_text(child, lines, indent + 1)


def _build_entailment_tree_json(sample) -> dict[str, Any] | None:
    """Convert a sample's record_nodes into a clean JSON entailment tree."""
    if not sample.record_nodes:
        return None

    tree = {}
    for node_id, node in sample.record_nodes.items():
        entry = {
            "content": node.content,
            "type": node.node_type.value,
            "parent": node.parent_id,
            "children": node.child_ids,
        }
        if node.is_observation and node.source_field is not None:
            entry["source_field"] = node.source_field
        tree[node_id] = entry

    return tree


# ---------------------------------------------------------------------------
# Train/eval split
# ---------------------------------------------------------------------------


def create_stratified_split(
    samples: list,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Create a stratified train/eval split by disease category.

    Args:
        samples: All ClinicalSample objects.
        test_ratio: Fraction for eval set (default 20%).
        seed: Random seed for reproducibility.

    Returns:
        (train_samples, eval_samples)
    """
    import random

    rng = random.Random(seed)

    # Group by disease category
    by_category: dict[str, list] = {}
    for s in samples:
        by_category.setdefault(s.disease_category, []).append(s)

    train = []
    test = []

    for cat, cat_samples in sorted(by_category.items()):
        # Shuffle within category
        shuffled = list(cat_samples)
        rng.shuffle(shuffled)

        # Split
        n_test = max(1, int(len(shuffled) * test_ratio))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    logger.info(
        f"Stratified split: {len(train)} train, {len(test)} eval "
        f"(from {len(samples)} total, {len(by_category)} categories)"
    )
    return train, test


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess supplementary training datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/supplementary"),
        help="Directory to save processed datasets (default: data/supplementary)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/.hf_cache"),
        help="HuggingFace cache directory (default: data/.hf_cache)",
    )
    parser.add_argument(
        "--direct-dir",
        type=Path,
        default=Path("mimic-iv-ext-direct-1.0.0"),
        help="Path to DiReCT dataset root (default: mimic-iv-ext-direct-1.0.0)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["medcase", "medical_o1", "direct_sft", "all"],
        default=["all"],
        help="Which datasets to download/process.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for testing).",
    )
    parser.add_argument(
        "--split-direct",
        action="store_true",
        help="Also create a train/eval split of DiReCT and generate SFT data.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    datasets_to_process = set(args.datasets)
    if "all" in datasets_to_process:
        datasets_to_process = {"medcase", "medical_o1", "direct_sft"}

    results = {}

    if "medcase" in datasets_to_process:
        logger.info("=" * 60)
        logger.info("Downloading MedCaseReasoning...")
        logger.info("=" * 60)
        try:
            counts = download_medcase_reasoning(
                output_dir=args.output_dir / "medcase_reasoning",
                cache_dir=args.cache_dir,
                max_samples=args.max_samples,
            )
            results["medcase_reasoning"] = counts
        except Exception as e:
            logger.error(f"Failed to download MedCaseReasoning: {e}", exc_info=True)
            results["medcase_reasoning"] = {"error": str(e)}

    if "medical_o1" in datasets_to_process:
        logger.info("=" * 60)
        logger.info("Downloading Medical-O1-Reasoning-SFT...")
        logger.info("=" * 60)
        try:
            counts = download_medical_o1(
                output_dir=args.output_dir / "medical_o1",
                cache_dir=args.cache_dir,
                max_samples=args.max_samples,
                language_filter="en",
            )
            results["medical_o1"] = counts
        except Exception as e:
            logger.error(
                f"Failed to download Medical-O1-Reasoning-SFT: {e}",
                exc_info=True,
            )
            results["medical_o1"] = {"error": str(e)}

    if "direct_sft" in datasets_to_process or args.split_direct:
        logger.info("=" * 60)
        logger.info("Generating DiReCT-format SFT training data...")
        logger.info("=" * 60)
        try:
            from src.data.reconcile import export
            report = export(args.direct_dir, args.output_dir / "direct_v2")
            results["direct_sft"] = {
                "train":report["sft_records"],
                "splits":{k:v["annotations"] for k,v in report["splits"].items()},
            }

        except Exception as e:
            logger.error(
                f"Failed to generate DiReCT SFT data: {e}", exc_info=True
            )
            results["direct_sft"] = {"error": str(e)}

    # Summary
    logger.info("=" * 60)
    logger.info("Download & preprocessing complete")
    logger.info("=" * 60)
    for name, info in results.items():
        logger.info(f"  {name}: {info}")

    # Save overall manifest
    manifest_path = args.output_dir / "manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")
    if any("error" in info for info in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
