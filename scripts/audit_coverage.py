"""Audit split metadata only; never run inference on final-test cases."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DiReCTDataset
from src.data.coverage import label_key, training_coverage, performance_by_coverage
from src.data.labels import normalize_diagnosis


def main():
    dataset = DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
    folder = ROOT/'data'/'splits_v2'
    splits = {name:json.loads((folder/f'{name}.json').read_text(encoding='utf-8'))
              for name in ['train','dev','test']}
    training, usable = training_coverage(folder)
    counts = {name:Counter(label_key(r['category'],r['pdd']) for r in rows)
              for name,rows in splits.items()}
    sft_rows = [json.loads(l) for l in (folder/'direct_train.jsonl').read_text(encoding='utf-8').splitlines()]
    sft_counts = Counter(label_key(r['disease_category'],r['pdd']) for r in sft_rows)
    table = []
    for category, pdd in sorted({(s.disease_category,s.pdd) for s in dataset}):
        key = label_key(category,pdd)
        kg = dataset.get_kg(category)
        kg_status = 'missing_kg' if kg is None else (
            'leaf' if normalize_diagnosis(pdd) in {normalize_diagnosis(n) for n in kg.get_all_pdd_names()}
            else 'internal' if normalize_diagnosis(pdd) in {normalize_diagnosis(n) for n in kg.get_all_node_names()}
            else 'unmapped_label')
        table.append({'category':category, 'diagnosis':pdd,
            **{name:counts[name][key] for name in splits},
            'training_sft':sft_counts[key], 'kg_coverage':kg_status})
    report = {'meaning':'Present/absent refers to project training data, not pretraining. No project fine-tuning has run.',
              'unique_labels':len(table), 'training_labels':len(training),
              'training_sft_labels':len(usable), 'labels':table, 'splits':{}, 'existing_runs':{}}
    for name,rows in splits.items():
        absent = [r for r in rows if label_key(r['category'],r['pdd']) not in training]
        report['splits'][name] = {'annotations':len(rows),
            'categories':len({r['category'] for r in rows}), 'labels':len(counts[name]),
            'absent_label_cases':len(absent),
            'absent_labels':dict(Counter(r['pdd'] for r in absent)),
            'missing_categories':sorted(set(dataset.categories)-{r['category'] for r in rows})}
    for name in ['dev_4b_corrected_routing_10','dev_8b_corrected_routing_10']:
        path = ROOT/'data'/'runs'/name/'predictions.jsonl'
        if path.exists():
            predictions = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
            report['existing_runs'][name] = performance_by_coverage(predictions,training,usable)
    out = ROOT/'data'/'audits'
    out.mkdir(parents=True,exist_ok=True)
    (out/'diagnosis_coverage.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'diagnosis_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    lines = ['# Diagnosis coverage audit', '', report['meaning'], '',
        f"There are {len(table)} diagnosis labels; {len(training)} occur in the training split and {len(usable)} in the usable DiReCT SFT export.", '',
        '| Split | Annotations | Categories | Labels | Cases with label absent from train |',
        '|---|---:|---:|---:|---:|']
    for name,r in report['splits'].items():
        lines.append(f"| {name} | {r['annotations']} | {r['categories']} | {r['labels']} | {r['absent_label_cases']} |")
    lines += ['', '## Labels absent from training', '', '| Diagnosis | Train | Dev | Test | KG |', '|---|---:|---:|---:|---|']
    for r in table:
        if not r['train']:
            lines.append(f"| {r['diagnosis']} | 0 | {r['dev']} | {r['test']} | {r['kg_coverage']} |")
    lines += ['', '## Existing small development runs', '', '| Run | Present-label correct / n | Absent-label correct / n |', '|---|---:|---:|']
    for name,groups in report['existing_runs'].items():
        p,a=groups['training_split']['present'],groups['training_split']['absent']
        lines.append(f"| {name} | {p['diagnosis_correct']}/{p['n']} | {a['diagnosis_correct']}/{a['n']} |")
    lines += ['', 'The absent-label case in each previous ten-case batch was Hypertrophic Cardiomyopathy. One success is not a dependable estimate of generalization.',
              '', 'Final-test labels were inspected for coverage only; no final-test predictions were generated. Splits remain frozen. See the CSV for all labels and KG gaps.',
              '', 'Three Gastritis labels are present in the training split but absent from usable SFT because no KG is supplied. This is tracked separately from labels absent from training entirely.',
              '', 'No project-trained model exists yet, so these strata describe data coverage rather than a learned seen/unseen effect.']
    (ROOT/'DIAGNOSIS_COVERAGE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['labels','existing_runs']},indent=2))


if __name__ == '__main__':
    main()
