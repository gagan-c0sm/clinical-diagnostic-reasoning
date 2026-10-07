"""Matched pipeline comparison with one GGUF base and switchable D3 adapter."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline.client import LocalClient
from src.training.budget import SharedBudget


def main():
    client=LocalClient()
    with SharedBudget('assessment',1200) as budget:
        try:
            for name,scale in [('base',0.0),('pilot',1.0)]:
                client.request('/lora-adapters',[{'id':0,'scale':scale}])
                active=client.request('/lora-adapters')
                if active[0]['scale']!=scale:raise ValueError('Adapter scale was not applied')
                output=ROOT/'data'/'runs'/f'd3_transfer_{name}_12'
                subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts'/'run_pipeline.py'),
                    '--limit','10','--include-absent-labels','--routing','single','--output',str(output)],
                    cwd=ROOT,check=True,timeout=max(1,budget.remaining()-60))
                (output/'adapter_scale.json').write_text(json.dumps({'scale':scale,'adapters':active},indent=2))
        finally:client.request('/lora-adapters',[{'id':0,'scale':0.0}])


if __name__=='__main__':main()
