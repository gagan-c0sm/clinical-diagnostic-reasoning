"""Small local qualitative generation check; no optimization or test-set use."""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HOME']=str(ROOT/'data'/'.hf_cache')


def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    from src.data import DiReCTDataset
    torch.set_num_threads(8);torch.manual_seed(42)
    base=ROOT/'data'/'models'/'Qwen3-4B-HF'
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    for module in (model.get_input_embeddings(),model.get_output_embeddings()):module.to(dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(model,ROOT/'data'/'training'/'d3_pilot_200'/'checkpoint-000200'/'adapter',local_files_only=True)
    model.eval();cases=[]
    validation_ids=set(json.loads((ROOT/'data'/'training'/'d3_pilot_200'/'status.json').read_text())['config']['validation_ids'])
    for l in (ROOT/'data'/'d3'/'val.jsonl').open(encoding='utf-8'):
        r=json.loads(l)
        if r['source'].endswith('MedCaseReasoning') and r['input_sha256'] not in validation_ids:
            cases.append({'id':r['input_sha256'],'source':'MedCaseReasoning validation outside monitoring subset',
                          'input':r['input'],'expected':r.get('final_diagnosis')})
            if len(cases)==2:break
    dataset=DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')
    rows=[json.loads(l) for l in (ROOT/'data'/'runs'/'coverage_single_12'/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    for category in ['Acute Coronary Syndrome','COPD']:
        r=next(r for r in rows if r['expected_category']==category);s=dataset.get_sample(r['sample_id'])
        cases.append({'id':s.sample_id,'source':'DiReCT development','input':s.inputs.get_all_text(),'expected':s.pdd})
    output=ROOT/'data'/'audits'/'d3_generation_check.jsonl'
    if output.exists():raise FileExistsError(output)
    for case in cases:
        messages=[{'role':'system','content':'Analyze clinical evidence. State a proposed diagnosis and at most three short evidence-to-conclusion steps. Distinguish uncertainty. Do not invent findings.'},
                  {'role':'user','content':case['input']}]
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt').cuda()
        results={}
        for adapted in [False,True]:
            from contextlib import nullcontext
            start=time.perf_counter()
            with (nullcontext() if adapted else model.disable_adapter()),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                generated=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=256,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
            continuation=generated[0,ids.shape[1]:]
            results['pilot' if adapted else 'base']={'text':tokenizer.decode(continuation,skip_special_tokens=True),
                'tokens':len(continuation),'hit_token_cap':len(continuation)==256,'seconds':time.perf_counter()-start}
        record={k:v for k,v in case.items() if k!='input'}|{'outputs':results}
        with output.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(record,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
