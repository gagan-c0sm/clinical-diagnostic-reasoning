"""Compare matched development cohorts and retain evidence about regressions."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('shortlist', type=Path)
    parser.add_argument('--output', type=Path, default=Path('ROUTING_CHECK.md'))
    args = parser.parse_args()
    runs = []
    for folder in (args.baseline,args.shortlist):
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if summary['split'] != 'dev':
            raise ValueError('This comparison is for development only')
        predictions = {r['sample_id']:r for r in (json.loads(l) for l in
            (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines())}
        runs.append((summary,predictions))
    if set(runs[0][1]) != set(runs[1][1]):
        raise ValueError('Cohorts differ')
    lines = ['# Category routing development check', '',
        'Same 4B model and development cohort, including all three diagnosis labels absent from project training. No fine-tuning has run.', '',
        '| Routing | Category correct | Diagnosis correct | Present-label diagnosis | Absent-label diagnosis | Mean seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for summary,predictions in runs:
        groups = summary['performance_by_label_coverage']['training_split']
        p,a=groups['present'],groups['absent']
        lines.append(f"| {summary['routing']} | {sum(r['category_correct'] for r in predictions.values())}/{len(predictions)} | "
            f"{sum(r['diagnosis_correct'] for r in predictions.values())}/{len(predictions)} | "
            f"{p['diagnosis_correct']}/{p['n']} | {a['diagnosis_correct']}/{a['n']} | {summary['mean_seconds']:.1f} |")
    fixed = sum(not r['diagnosis_correct'] and runs[1][1][sid]['diagnosis_correct'] for sid,r in runs[0][1].items())
    harmed = sum(r['diagnosis_correct'] and not runs[1][1][sid]['diagnosis_correct'] for sid,r in runs[0][1].items())
    lines += ['',f'Shortlist fixed {fixed} baseline diagnosis errors and lost {harmed} baseline successes.', '',
        f"Expected-category coverage in the shortlist: {runs[1][0]['candidate_category_recall']:.1%}.", '',
        '## Changed diagnoses', '', '| Expected diagnosis | Single result | Shortlist result |', '|---|---|---|']
    for sid,r in runs[0][1].items():
        other = runs[1][1][sid]
        if r['diagnosis'] != other['diagnosis']:
            lines.append(f"| {r['expected_diagnosis']} | {r['diagnosis']} ({r['status']}) | {other['diagnosis']} ({other['status']}) |")
    lines += ['', 'This is a small development comparison. Labels absent from project training may still be familiar from pretraining; their KG criteria are supplied. Final-test predictions were not generated.']
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
