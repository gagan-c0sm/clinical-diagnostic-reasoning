"""Broader matched development comparison, charged to the shared budget."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.budget import SharedBudget,atomic_json
from src.pipeline.client import LocalClient


def main():
    with SharedBudget('assessment',1200) as budget:
        client=LocalClient()
        try:
            for name,scale in [('base',0.),('d4',1.)]:
                client.request('/lora-adapters',[{'id':0,'scale':scale}])
                if client.request('/lora-adapters')[0]['scale']!=scale:raise ValueError('Adapter activation mismatch')
                output=ROOT/'data'/'runs'/f'broad_{name}_dev'
                subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts'/'run_pipeline.py'),
                    '--limit','24','--include-absent-labels','--output',str(output)],
                    cwd=ROOT,check=True,timeout=max(1,budget.remaining()-60))
                atomic_json(output/'adapter.json',{'scale':scale,'adapters':client.request('/lora-adapters')})
        finally:client.request('/lora-adapters',[{'id':0,'scale':0.}])


if __name__=='__main__':main()
