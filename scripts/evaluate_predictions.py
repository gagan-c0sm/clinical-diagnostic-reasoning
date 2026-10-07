"""D6: score saved predictions offline without generating or training a model."""
import argparse,hashlib,json,statistics,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import DiReCTDataset
from src.data.labels import normalize_diagnosis
from src.evaluation.embeddings import BiomedicalEncoder
from src.evaluation.metrics import score_case,prf


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('runs',nargs='+',type=Path,help='Completed run directories')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--threshold',type=float,default=.8)
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if not 0.<args.threshold<=1.:parser.error('Threshold must be in (0,1]')
    if args.output.exists():parser.error('Output directory already exists; preserve previous results')
    import torch
    torch.set_num_threads(4)
    split_path=ROOT/'data'/'splits_v2'/f'{args.split}.json'
    allowed={r['sample_id'] for r in json.loads(split_path.read_text())}
    loaded=[]
    for folder in args.runs:
        summary=json.loads((folder/'summary.json').read_text())
        rows=[json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        ids=[r['sample_id'] for r in rows]
        if (summary['split']!=args.split or summary['n']!=len(rows) or not rows or
            len(ids)!=len(set(ids)) or not set(ids)<=allowed):
            raise ValueError(f'Incomplete run, duplicate IDs or wrong split: {folder}')
        loaded.append((folder,rows))
    started=time.monotonic()
    encoder=BiomedicalEncoder(ROOT/'data'/'models'/'S-PubMedBert-MS-MARCO',args.device)
    dataset=DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
    args.output.mkdir(parents=True)
    combined={}
    for folder,rows in loaded:
        if folder.name in combined:raise ValueError('Run names must be unique')
        scores=[]
        for i,row in enumerate(rows):
            scores.append(score_case(row,dataset.get_sample(row['sample_id']),encoder,args.threshold))
            print(f'{folder.name}: {i+1}/{len(rows)}',flush=True)
        metrics={}
        for key in ['observations_exact','observations_semantic','explanation_steps','diagnostic_milestones']:
            counts={k:sum(s[key][k] for s in scores) for k in ['matched','predicted','reference']}
            metrics[key]={'micro':prf(**counts),
                'macro':{k:statistics.mean(s[key][k] for s in scores) for k in ['precision','recall','f1']}}
        summary={'n':len(scores),'diagnosis_accuracy':statistics.mean(s['diagnosis_correct'] for s in scores),
            'diagnosis_accuracy_annotation_root':statistics.mean(s['diagnosis_annotation_root_correct'] for s in scores),
            'reference_label_disagreements':sum(normalize_diagnosis(s['reference_metadata_pdd'])!=normalize_diagnosis(s['reference_annotation_root']) for s in scores),
            'category_accuracy':statistics.mean(s['category_correct'] for s in scores),
            'status_counts':dict(Counter(s['status'] for s in scores)),
            'tree_error_cases':sum(bool(s['tree_errors']) for s in scores),
            'grounded_observations':sum(s['grounded_observations'] for s in scores),
            'metrics':metrics,'prediction_sha256':hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest(),
            'sample_ids':[s['sample_id'] for s in scores]}
        combined[folder.name]=summary
        (args.output/f'{folder.name}.cases.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'split':args.split,'threshold':args.threshold,'threshold_calibrated':False,
        'encoder':encoder.manifest,'pooling':'attention-masked mean, full-text nonoverlapping chunks, L2 normalization',
        'chunked_unique_texts':encoder.chunked_texts,'unique_encoded_texts':len(encoder.cache),
        'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),*sorted((ROOT/'src'/'evaluation').glob('*.py'))]},
        'seconds':time.monotonic()-started,'device':args.device,'runs':combined,
        'note':'Local annotation agreement proxies. Uncalibrated similarity is not clinical correctness or an exact reproduction of DiReCT paper metrics.'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Local annotation evaluation','',f'Split: {args.split}. Similarity threshold: {args.threshold} (uncalibrated).',
        '',result['note'],'','| Run | Metadata PDD | Annotated final diagnosis | Semantic evidence P/R/F1 | Explanation P/R/F1 | Milestone P/R/F1 |',
        '|---|---:|---:|---|---|---|']
    def fmt(m):return '/'.join(f"{m[k]:.3f}" for k in ['precision','recall','f1'])
    for name,s in combined.items():
        m=s['metrics'];lines.append(f"| {name} | {s['diagnosis_accuracy']:.1%} | {s['diagnosis_accuracy_annotation_root']:.1%} | {fmt(m['observations_semantic']['micro'])} | {fmt(m['explanation_steps']['micro'])} | {fmt(m['diagnostic_milestones']['micro'])} |")
    lines+=['','P/R/F1 above are micro scores. Macro scores, counts, hashes and settings are in summary.json.',
        'Matched evidence and explanation pairs are in each *.cases.json file.',
        'Evidence matches require the same source field and a verbatim-grounded prediction.',
        'Explanation matches require the same normalized milestone label, similar rationale, and at least one similar evidence pair.',
        'All cases, including abstentions, wrong categories and missing KGs, remain in the denominators.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Evaluation saved:',args.output,flush=True)


if __name__=='__main__':main()
