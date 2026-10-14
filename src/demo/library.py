"""Development-only case library. Annotation answers never enter case inputs."""
import copy,hashlib,json
from functools import lru_cache
from src.data import DiReCTDataset
from src.demo.runner import ROOT


@lru_cache(maxsize=1)
def records():
    allowed={r['sample_id'] for r in json.loads((ROOT/'data/splits_v2/dev.json').read_text(encoding='utf-8'))}
    dataset=DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
    saved={r['sample_id']:r for r in map(json.loads,(ROOT/'data/runs/broad_d4_dev/predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    output={}
    for sample in sorted(dataset,key=lambda s:s.sample_id):
        if sample.sample_id not in allowed:continue
        key=hashlib.sha256(sample.sample_id.encode()).hexdigest()[:16]
        case={'case_id':f'Example case {len(output)+1:02d}',**{f'input{i}':sample.inputs.get_field(i) for i in range(1,7)}}
        symptom=case['input1'].strip()
        output[key]={'id':key,'title':symptom[:110] if symptom and symptom.lower()!='n/a' else 'Explore a clinical record',
            'category':sample.disease_category,'label':case['case_id'],
            'replay':sample.sample_id in saved,'case':case,'saved':saved.get(sample.sample_id)}
    return output


def catalog():
    return [{k:v for k,v in r.items() if k not in ('case','saved')} for r in records().values()]


def case_for(key):
    return copy.deepcopy(records()[key]['case'])


def saved_for(key):
    saved=records()[key]['saved']
    if not saved:raise ValueError('This case has no saved D4 run. Load its input to run a new analysis.')
    # Explicit output allowlist excludes ground truth and evaluation labels.
    result={k:copy.deepcopy(saved[k]) for k in ('status','category','diagnosis','nodes','observations','trace','seconds','calls','warnings') if k in saved}
    result.update(model_choice='d4',case_id=records()[key]['label'],replay=True,
                  provenance='Saved D4 run; original diagnosis unchanged.')
    return result
