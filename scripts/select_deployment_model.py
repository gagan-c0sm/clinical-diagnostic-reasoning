"""Select a deployment candidate from the completed matched development runs."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import DiReCTDataset
from src.data.labels import normalize_diagnosis
from src.pipeline.schema import validate_tree
from src.training.budget import atomic_json


def main():
    runs={};dataset=DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
    for name,scale in [('base',0.),('d4',1.)]:
        folder=ROOT/'data'/'runs'/f'broad_{name}_dev'
        summary=json.loads((folder/'summary.json').read_text())
        adapter=json.loads((folder/'adapter.json').read_text())
        rows=[json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        if summary['split']!='dev' or len(rows)!=summary['n'] or adapter['scale']!=scale:
            raise ValueError('Incomplete or mismatched evaluation metadata')
        if not adapter['adapters'] or any(a['scale']!=scale for a in adapter['adapters']):
            raise ValueError('Adapter activation metadata mismatch')
        if any(r['status']=='error' for r in rows):raise ValueError('Resolve runtime errors before selecting a model')
        for row in rows:
            if row['nodes']:
                errors=validate_tree(row['nodes'],dataset.get_sample(row['sample_id']).inputs)
                if errors:raise ValueError((name,row['sample_id'],errors))
        runs[name]={'summary':summary,'rows':rows,'path':str(folder.relative_to(ROOT))}
    base,d4=runs['base'],runs['d4']
    if [r['sample_id'] for r in base['rows']]!=[r['sample_id'] for r in d4['rows']]:
        raise ValueError('Cohorts differ')
    for key in ['source_sha256','split_sha256','routing']:
        if base['summary'][key]!=d4['summary'][key]:raise ValueError(f'Unmatched {key}')
    def is_correct(row):
        sample=dataset.get_sample(row['sample_id'])
        return normalize_diagnosis(row.get('diagnosis') or '')==normalize_diagnosis(sample.root_node.content)
    correct={name:sum(is_correct(r) for r in run['rows']) for name,run in runs.items()}
    selected='d4' if correct['d4']>correct['base'] else 'base'
    result={'selected':selected,'utc_selected':datetime.now(timezone.utc).isoformat(),
        'n':len(base['rows']),'diagnosis_correct':correct,
        'runs':{name:run['path'] for name,run in runs.items()},
        'metadata_pdd_correct':{name:sum(r['diagnosis_correct'] for r in run['rows']) for name,run in runs.items()},
        'diagnosis_reference':'annotation_tree_root',
        'd4_fixes':sum(not is_correct(b) and is_correct(d) for b,d in zip(base['rows'],d4['rows'])),
        'd4_regressions':sum(is_correct(b) and not is_correct(d) for b,d in zip(base['rows'],d4['rows'])),
        'adapter':'data/models/d4_best.gguf','best_step':400,
        'selection_rule':'Higher matched annotation-root diagnosis count; ties retain base. D4 deployment authorized by user.',
        'note':'Development selection only. Clinical rationale correctness and final-test accuracy are unverified.'}
    atomic_json(ROOT/'data'/'models'/'selected_model.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
