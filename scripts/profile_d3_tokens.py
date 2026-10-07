"""Measure complete chat examples; never silently truncate training targets."""
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]


def encode(record, tokenizer):
    messages=[{'role':'system','content':record['instruction']},
              {'role':'user','content':record['input']}]
    prefix=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    # Qwen's non-thinking assistant prefix includes an empty think block.
    prompt=tokenizer.encode(prefix,add_special_tokens=False)
    completion=tokenizer.encode(record['output']+'<|im_end|>',add_special_tokens=False)
    return prompt+completion, [-100]*len(prompt)+completion


def main():
    folder=ROOT/'data'/'d3'
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data'/'models'/'Qwen3-4B-HF',local_files_only=True)
    report={'policy':'Full input and full output; no truncation. Tokenized file retains examples <=2048.', 'splits':{}}
    for split in ['train','val','test']:
        lengths=[]
        target_lengths=[]
        output=folder/f'{split}_tokens.jsonl'
        with output.open('w',encoding='utf-8') as out:
            for line in (folder/f'{split}.jsonl').open(encoding='utf-8'):
                r=json.loads(line);ids,labels=encode(r,tokenizer)
                lengths.append(len(ids));target_lengths.append(sum(x!=-100 for x in labels))
                if len(ids)<=2048:
                    out.write(json.dumps({'input_ids':ids,'labels':labels,'source':r['source'],
                        'input_sha256':r['input_sha256']})+'\n')
        lengths.sort()
        report['splits'][split]={'records':len(lengths), 'p50':lengths[int(len(lengths)*.5)],
            'p90':lengths[int(len(lengths)*.9)],'p95':lengths[int(len(lengths)*.95)],'max':max(lengths),
            'fit':{str(n):sum(x<=n for x in lengths) for n in [512,1024,1536,2048]},
            'mean_target_tokens':sum(target_lengths)/len(target_lengths)}
        print(split,report['splits'][split],flush=True)
    (folder/'token_profile.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
