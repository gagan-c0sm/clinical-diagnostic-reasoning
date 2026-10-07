"""Resume D4, then evaluate its best adapter within the same shared budget."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.budget import SharedBudget,atomic_json
from src.pipeline.client import LocalClient
from src.pipeline.schema import validate_tree
from src.data import DiReCTDataset


def main():
    run=ROOT/'data'/'training'/'d4_session_600'
    latest=json.loads((run/'latest.json').read_text())['checkpoint']
    subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts'/'train_d3_session.py'),
        '--output',str(run),'--resume',str(run/latest),'--data-dir',str(ROOT/'data'/'d4_compact'),
        '--steps','600','--grad-accum','2','--val-every','50','--val-samples','30',
        '--max-hours','1.6666667','--lr','0.00005','--budget-stage','d4'],cwd=ROOT,check=True)
    status=json.loads((run/'status.json').read_text())
    if status['best_checkpoint']=='checkpoint-000000':
        atomic_json(run/'selection.json',{'selected':'base','reason':'D4 validation did not improve over initial checkpoint'})
        return
    best=json.loads((run/'best.json').read_text())
    adapter=run/best['adapter']
    converted=ROOT/'data'/'models'/'d4_best.gguf'
    output=ROOT/'data'/'runs'/'d4_transfer_12'
    with SharedBudget('assessment',1200) as budget:
        def execute(command):
            subprocess.run(command,cwd=ROOT,check=True,timeout=max(1,budget.remaining()-30))
        execute([sys.executable,'-B',str(ROOT/'tools'/'llama-converter'/'convert_lora_to_gguf.py'),
            '--base',str(ROOT/'data'/'models'/'Qwen3-4B-HF'),'--outfile',str(converted),'--outtype','f16',str(adapter)])
        import shutil
        pwsh=shutil.which('pwsh')
        if not pwsh:raise RuntimeError('PowerShell runtime unavailable')
        try:
            execute([pwsh,'-NoProfile','-File',str(ROOT/'scripts'/'start_local_model.ps1'),
                '-ModelSize','4B','-AdapterPath',str(converted)])
            client=LocalClient();client.request('/lora-adapters',[{'id':0,'scale':1.0}])
            if client.request('/lora-adapters')[0]['scale']!=1.0:raise ValueError('Adapter activation failed')
            execute([sys.executable,'-u','-B',str(ROOT/'scripts'/'run_pipeline.py'),
                '--limit','10','--include-absent-labels','--output',str(output)])
        finally:
            subprocess.run([pwsh,'-NoProfile','-File',str(ROOT/'scripts'/'stop_local_model.ps1')],cwd=ROOT,timeout=20)
        dataset=DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
        candidates=[json.loads(l) for l in (output/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        baseline=[json.loads(l) for l in (ROOT/'data'/'runs'/'d3_transfer_base_12'/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        if {r['sample_id'] for r in candidates}!={r['sample_id'] for r in baseline}:raise ValueError('Cohort mismatch')
        for r in candidates:
            sample=dataset.get_sample(r['sample_id'])
            if r['nodes'] and validate_tree(r['nodes'],sample.inputs):raise ValueError('Output validation failed')
        score=sum(r['diagnosis_correct'] for r in candidates);base_score=sum(r['diagnosis_correct'] for r in baseline)
        decision={'selected':'d4_candidate' if score>base_score else 'base',
            'baseline_correct':base_score,'d4_correct':score,'n':len(candidates),
            'adapter':str(adapter),'gguf':str(converted),'validation_best':status['best_validation_loss'],
            'note':'Provisional development selection only. Medical explanation correctness and final-test accuracy are unverified.'}
        atomic_json(run/'selection.json',decision)
        print('Selection:',decision,flush=True)


if __name__=='__main__':main()
