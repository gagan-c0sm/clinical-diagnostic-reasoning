"""Deterministic patient-grouped train/dev/test export; no model calls."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .download import generate_direct_sft_data
from .labels import normalize_diagnosis
from .loader import DiReCTDataset


def grouped_split(samples, seed=42):
    """Approximate 70/15/15, preserving patient groups and category coverage.

    Choose the best of deterministic shuffled group allocations. Tiny classes
    cannot always occur in all three splits; report actual counts instead.
    """
    groups = defaultdict(list)
    for s in samples:
        groups[s.patient_id].append(s)
    totals = Counter(s.disease_category for s in samples)
    fractions = {"train": .70, "dev": .15, "test": .15}
    rng = random.Random(seed)
    best = None
    for _ in range(500):
        ids = sorted(groups)
        rng.shuffle(ids)
        n = len(ids)
        partitions = {"train": ids[:int(n*.70)],
                      "dev": ids[int(n*.70):int(n*.85)],
                      "test": ids[int(n*.85):]}
        candidate = {k: [s for g in v for s in groups[g]] for k,v in partitions.items()}
        score = 0
        for name, items in candidate.items():
            counts = Counter(s.disease_category for s in items)
            for cat, total in totals.items():
                target = total * fractions[name]
                score += (counts[cat]-target)**2 / max(target, 1)
                if not counts[cat] and (name == "train" or total >= 7):
                    score += 30
        if best is None or score < best[0]:
            best = score, candidate
    return {k: sorted(v, key=lambda s: s.sample_id) for k,v in best[1].items()}


def export(root, output):
    dataset = DiReCTDataset(root)
    splits = grouped_split(dataset.samples)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen splits: {output}")
    output.mkdir(parents=True)
    report = {"version": 2, "seed": 42, "grouping": "patient_id",
              "policy": "Tune on dev only; reserve test for final assessment.",
              "schema": "flat nodes: content/type/parent/children/source_field",
              "missing_kg_policy": "abstain from KG reasoning; report coverage",
              "splits": {}}
    for name, samples in splits.items():
        rows = [{"sample_id":s.sample_id, "note_id":s.note_id,
                 "patient_id":s.patient_id, "category":s.disease_category,
                 "pdd":s.pdd} for s in samples]
        (output/f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        report["splits"][name] = {"annotations":len(samples),
            "patients":len({s.patient_id for s in samples}),
            "categories":dict(sorted(Counter(s.disease_category for s in samples).items()))}
    report["sft_records"] = generate_direct_sft_data(
        splits["train"], dataset.knowledge_graphs, output/"direct_train.jsonl")
    report["unresolved_labels"] = dict(Counter(
        f"{s.disease_category}: {s.pdd}" for s in dataset
        if (kg := dataset.get_kg(s.disease_category)) and
        normalize_diagnosis(s.pdd) not in {normalize_diagnosis(n) for n in kg.get_all_node_names()}))
    report["direct_observation_diagnosis_edges"] = sum(
        1 for s in dataset for n in s.observations
        if n.parent_id and s.record_nodes[n.parent_id].is_diagnosis)
    report["source_sha256"] = {s.sample_id:hashlib.sha256(s.file_path.read_bytes()).hexdigest() for s in dataset}
    (output/"manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
