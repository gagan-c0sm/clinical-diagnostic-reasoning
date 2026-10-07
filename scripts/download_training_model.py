"""Download the pinned official HF checkpoint, verifying weight hashes."""
import hashlib
import json
import urllib.request
from pathlib import Path
from setup_local_runtime import fetch, get_json, ROOT

REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'


def main():
    repo = 'Qwen/Qwen3-4B'
    files = get_json(f'https://huggingface.co/api/models/{repo}/tree/{REVISION}')
    folder = ROOT/'data'/'models'/'Qwen3-4B-HF'
    folder.mkdir(parents=True,exist_ok=True)
    manifest = {'repo':repo,'revision':REVISION,'sha256':{}}
    for item in files:
        name=item['path']
        if not name.endswith(('.safetensors','.json','.txt','.jinja')): continue
        target=folder/name
        url=f'https://huggingface.co/{repo}/resolve/{REVISION}/{name}'
        if 'lfs' in item:
            digest=fetch(url,target,item['lfs']['oid'])
        else:
            if not target.exists():
                with urllib.request.urlopen(url,timeout=60) as response:
                    target.write_bytes(response.read())
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest['sha256'][name]=digest
    (folder/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Trainable checkpoint verified',flush=True)


if __name__=='__main__': main()
