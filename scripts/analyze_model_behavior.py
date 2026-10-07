"""Decompose the frozen broad comparison without new inference or training."""
import json,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import DiReCTDataset
from src.data.labels import normalize_diagnosis as norm
from src.evaluation.metrics import prf


def main():
    dataset=DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
    predictions={};scores={}
    for name in ['base','d4']:
        folder=ROOT/'data'/'runs'/f'broad_{name}_dev'
        predictions[name]=[json.loads(x) for x in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        scores[name]=json.loads((ROOT/'data'/'evaluation'/'d6_dev'/f'broad_{name}_dev.cases.json').read_text(encoding='utf-8'))
    ids=[r['sample_id'] for r in predictions['base']]
    assert ids==[r['sample_id'] for r in predictions['d4']]
    for name in scores:assert ids==[r['sample_id'] for r in scores[name]]
    shared=[i for i in range(len(ids)) if all(predictions[n][i]['category_correct'] for n in predictions)]
    cases=[];summary={}
    for i,sample_id in enumerate(ids):
        sample=dataset.get_sample(sample_id)
        root=sample.root_node.content
        row={'case':i+1,'sample_id':sample_id,'expected_category':sample.disease_category,
            'metadata_pdd':sample.pdd,'annotation_root':root,'reference_disagreement':norm(root)!=norm(sample.pdd)}
        for name in predictions:
            p=predictions[name][i];correct=norm(p.get('diagnosis') or '')==norm(root)
            row[name]={'category':p['category'],'diagnosis':p['diagnosis'],'status':p['status'],
                'category_correct':p['category_correct'],'metadata_correct':p['diagnosis_correct'],'root_correct':correct,
                'failure_stage':('none' if correct else 'classification_abstention' if p['category']=='Uncertain'
                    else 'category_routing' if not p['category_correct'] else 'reasoning_or_subtype'),
                'classification_summary':p['classification_evidence'],'trace':p['trace'],
                'last_step_evidence':[o for o in p['observations'] if p['trace'] and o['id'] in p['trace'][-1]['evidence_ids']]}
        cases.append(row)
    for name,rows in predictions.items():
        traces=[t for r in rows for t in r['trace']]
        stages={}
        for stage in ['classify','extract','reason']:
            calls=[c for r in rows for c in r['calls'] if c['stage'].startswith(stage)]
            stages[stage]={'calls':len(calls),'seconds':sum(c['seconds'] for c in calls),
                'completion_tokens':sum(c['usage'].get('completion_tokens',0) for c in calls)}
        aggregates={}
        for group,indices in [('all',range(len(ids))),('both_category_correct',shared)]:
            aggregates[group]={key:prf(**{count:sum(scores[name][i][key][count] for i in indices)
                for count in ['matched','predicted','reference']}) for key in ['observations_semantic','explanation_steps']}
        summary[name]={'metadata_diagnoses_correct':sum(c[name]['metadata_correct'] for c in cases),
            'annotation_root_diagnoses_correct':sum(c[name]['root_correct'] for c in cases),
            'categories_correct':sum(c[name]['category_correct'] for c in cases),
            'status_counts':dict(Counter(r['status'] for r in rows)),
            'incorrect_complete_root':sum(c[name]['status']=='complete' and not c[name]['root_correct'] for c in cases),
            'failure_stages':dict(Counter(c[name]['failure_stage'] for c in cases)),
            'mean_seconds':statistics.mean(r['seconds'] for r in rows),
            'mean_classification_words':statistics.mean(len(r['classification_evidence'].split()) for r in rows),
            'mean_reasoning_words':statistics.mean(len(t['rationale'].split()) for t in traces),
            'mean_evidence_ids_per_decision':statistics.mean(len(t['evidence_ids']) for t in traces),
            'single_evidence_decisions':sum(len(t['evidence_ids'])==1 for t in traces),
            'reasoning_decisions':len(traces),'stop_decisions':sum(t['choice']=='STOP' for t in traces),
            'selected_observations':sum(len(r['observations']) for r in rows),
            'stages':stages,'metrics':aggregates}
    training=[json.loads(x) for x in (ROOT/'data'/'d4_compact'/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    targets=[json.loads(x['output']) for x in training if x['task_type']=='reason']
    result={'n':len(ids),'both_category_correct_n':len(shared),'summary':summary,'cases':cases,
        'paired_outcomes':dict(Counter('both_correct' if c['base']['root_correct'] and c['d4']['root_correct']
            else 'base_only' if c['base']['root_correct'] else 'd4_only' if c['d4']['root_correct'] else 'both_wrong' for c in cases)),
        'training_reason_targets':{'n':len(targets),'evidence_cardinality':dict(Counter(len(t['evidence_ids']) for t in targets)),
            'stop_targets':sum(t['choice']=='STOP' for t in targets),
            'mean_rationale_words':statistics.mean(len(t['rationale'].split()) for t in targets)},
        'note':'Saved development-output audit. Annotation-root and metadata labels are reported separately. No model/default changed.'}
    folder=ROOT/'data'/'evaluation'/'behavior_audit';folder.mkdir(exist_ok=True)
    (folder/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['n','paired_outcomes','training_reason_targets']},indent=2))


if __name__=='__main__':main()
