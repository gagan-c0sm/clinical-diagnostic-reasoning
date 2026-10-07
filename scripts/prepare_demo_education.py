"""Cache teaching text for saved development outputs without rerunning diagnosis."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.demo.library import records,saved_for
from src.demo.education import explain,teaching_adapter
from src.demo.runner import model_lock
from src.pipeline.client import LocalClient

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=25);p.add_argument('--port',type=int,default=8081);p.add_argument('--retry-partial',action='store_true');a=p.parse_args()
    folder=ROOT/'data/demo_education';folder.mkdir(parents=True,exist_ok=True)
    client=LocalClient(a.port)
    with model_lock():
        adapters=client.request('/lora-adapters')
        if len(adapters)!=1 or Path(adapters[0]['path']).resolve()!=ROOT/'data/models/d4_best.gguf' or adapters[0]['scale']!=1:
            raise RuntimeError('D4 must be active')
        count=0
        for key,record in records().items():
            if not record['replay']:continue
            target=folder/f'{key}.json'
            if target.exists():
                cached=json.loads(target.read_text(encoding='utf-8'))
                if cached.get('version')==2 and (not a.retry_partial or cached['mode']=='generated'):continue
            if count>=a.limit:break
            print(f'Explaining {record["label"]} / {record["category"]}',flush=True)
            with teaching_adapter(client):result=explain(saved_for(key),client)
            target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'  {len(result["steps"])} steps, {result["seconds"]:.1f}s, errors: {result["errors"]}',flush=True)
            count+=1
