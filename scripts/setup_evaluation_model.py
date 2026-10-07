"""Fetch the pinned biomedical encoder into the project; no global cache."""
import hashlib,json,urllib.request
from setup_local_runtime import ROOT,fetch,get_json

REPO='pritamdeka/S-PubMedBert-MS-MARCO'
REVISION='96786c7024f95c5aac7f2b9a18086c7b97b23036'
FILES=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json',
       'special_tokens_map.json','vocab.txt','sentence_bert_config.json',
       '1_Pooling/config.json','modules.json','README.md']


def main():
    folder=ROOT/'data'/'models'/'S-PubMedBert-MS-MARCO'
    folder.mkdir(parents=True,exist_ok=True)
    manifest={'repo':REPO,'revision':REVISION,'license':'cc-by-nc-2.0','sha256':{}}
    for name in FILES:
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        if name=='pytorch_model.bin':
            tree=get_json(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}')
            item=next(x for x in tree if x['path']==name)
            digest=fetch(url,target,item['lfs']['oid'])
        else:
            if not target.exists():
                with urllib.request.urlopen(url,timeout=60) as response:target.write_bytes(response.read())
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
        manifest['sha256'][name]=digest
    (folder/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Project-local evaluation model ready:',folder,flush=True)


if __name__=='__main__':main()
