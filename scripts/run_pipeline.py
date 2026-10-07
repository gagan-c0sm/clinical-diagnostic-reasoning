"""Run a small development batch; final-test access requires explicit --split test."""
import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DiReCTDataset
from src.data.labels import normalize_diagnosis
from src.data.coverage import label_key, training_coverage, performance_by_coverage
from src.pipeline.client import LocalClient
from src.pipeline.pipeline import DiagnosticPipeline


def select_balanced(rows, limit):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['category']].append(row)
    selected = []
    while buckets and len(selected) < limit:
        for cat in list(sorted(buckets)):
            if len(selected) >= limit:
                break
            selected.append(buckets[cat].pop(0))
            if not buckets[cat]:
                del buckets[cat]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--routing', choices=['single','shortlist'], default='single')
    parser.add_argument('--include-absent-labels', action='store_true',
                        help='Add all development cases whose labels are absent from train to the selected batch')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    if args.include_absent_labels and args.split != 'dev':
        parser.error('--include-absent-labels is a development-only diagnostic')
    client = LocalClient(args.port)
    client.request('/health')
    models = client.request('/v1/models')
    dataset = DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
    split_path = ROOT/'data'/'splits_v2'/f'{args.split}.json'
    all_rows = json.loads(split_path.read_text(encoding='utf-8'))
    rows = select_balanced(all_rows, args.limit)
    training_labels, usable_labels = training_coverage(ROOT/'data'/'splits_v2')
    if args.include_absent_labels:
        selected_ids = {r['sample_id'] for r in rows}
        rows += [r for r in all_rows if r['sample_id'] not in selected_ids and
                 label_key(r['category'],r['pdd']) not in training_labels]
    out = args.output or ROOT/'data'/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    pipeline = DiagnosticPipeline(client, dataset.knowledge_graphs, dataset.categories, routing=args.routing)
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['src/pipeline','src/data'] for p in (ROOT/folder).glob('*.py')}
    runtime_manifest = json.loads((ROOT/'tools'/'llama.cpp'/'manifest.json').read_text(encoding='utf-8'))
    results = []
    for index, row in enumerate(rows):
        sample = dataset.get_sample(row['sample_id'])
        result = pipeline.run(sample.inputs)
        # Ground truth is attached only after inference, for local error analysis.
        result.update({'sample_id':sample.sample_id,
                       'expected_category':sample.disease_category,
                       'expected_diagnosis':sample.pdd})
        result['category_correct'] = result['category'] == sample.disease_category
        result['diagnosis_correct'] = (normalize_diagnosis(result['diagnosis'] or '') == normalize_diagnosis(sample.pdd))
        key = label_key(sample.disease_category,sample.pdd)
        result['label_present_in_training_split'] = key in training_labels
        result['label_present_in_training_sft'] = key in usable_labels
        result['initial_category_correct'] = result.get('initial_category') == sample.disease_category
        result['candidate_contains_expected_category'] = sample.disease_category in result.get('category_candidates',[])
        predicted = {(o['source_field'], ' '.join(o['text'].split()).casefold()) for o in result['observations']}
        reference = {(o.source_field, ' '.join(o.content.split()).casefold()) for o in sample.observations}
        overlap = len(predicted & reference)
        result['observation_exact_precision'] = overlap/len(predicted) if predicted else 0
        result['observation_exact_recall'] = overlap/len(reference) if reference else 0
        results.append(result)
        with (out/'predictions.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
        print(f"{index+1}/{len(rows)} {sample.disease_category}: {result['status']}, "
              f"category_ok={result['category_correct']}, diagnosis_ok={result['diagnosis_correct']}, "
              f"{result['seconds']:.1f}s", flush=True)
    summary = {'split':args.split, 'n':len(results), 'models':models, 'routing':args.routing,
               'include_absent_labels':args.include_absent_labels,
               'performance_by_label_coverage':performance_by_coverage(results,training_labels,usable_labels),
               'candidate_category_recall':statistics.mean(r['candidate_contains_expected_category'] for r in results),
               'initial_category_accuracy':statistics.mean(r['initial_category_correct'] for r in results),
               'routing_corrections':sum(not r['initial_category_correct'] and r['category_correct'] for r in results),
               'routing_regressions':sum(r['initial_category_correct'] and not r['category_correct'] for r in results),
               'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
               'status_counts':dict(Counter(r['status'] for r in results)),
               'category_accuracy':statistics.mean(r['category_correct'] for r in results),
               'diagnosis_accuracy_alias_exact':statistics.mean(r['diagnosis_correct'] for r in results),
               'mean_seconds':statistics.mean(r['seconds'] for r in results),
               'median_seconds':statistics.median(r['seconds'] for r in results),
               'observation_exact_precision':statistics.mean(r['observation_exact_precision'] for r in results),
               'observation_exact_recall':statistics.mean(r['observation_exact_recall'] for r in results),
               'runtime_manifest':runtime_manifest,
               'note':'Development diagnostics, not paper metrics. Exact observation scores penalize equivalent wording/span boundaries.',
               'source_sha256':source_hashes}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Results:', out, flush=True)


if __name__ == '__main__':
    main()
