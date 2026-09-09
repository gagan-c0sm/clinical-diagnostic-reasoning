"""Label coverage in project data, not claims about model pretraining exposure."""
import json
from collections import Counter
from pathlib import Path

from .labels import normalize_diagnosis


def label_key(category, diagnosis):
    return category, normalize_diagnosis(diagnosis)


def training_coverage(split_dir):
    folder = Path(split_dir)
    rows = json.loads((folder/'train.json').read_text(encoding='utf-8'))
    labels = {label_key(r['category'], r['pdd']) for r in rows}
    sft = [json.loads(line) for line in (folder/'direct_train.jsonl').read_text(encoding='utf-8').splitlines()]
    usable = {label_key(r['disease_category'], r['pdd']) for r in sft}
    return labels, usable


def performance_by_coverage(predictions, labels, usable):
    result = {}
    for name, pool in [('training_split', labels), ('training_sft', usable)]:
        groups = {}
        for present in (True, False):
            selected = [r for r in predictions if
                (label_key(r['expected_category'], r['expected_diagnosis']) in pool) == present]
            n = len(selected)
            groups['present' if present else 'absent'] = {
                'n':n, 'category_correct':sum(bool(r['category_correct']) for r in selected),
                'diagnosis_correct':sum(bool(r['diagnosis_correct']) for r in selected),
                'diagnosis_accuracy':sum(bool(r['diagnosis_correct']) for r in selected)/n if n else None,
                'status_counts':dict(Counter(r['status'] for r in selected))}
        result[name] = groups
    return result
