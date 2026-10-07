"""Bounded, resumable local D3 sessions. No remote model or data access."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'data'/'.hf_cache')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--grad-accum',type=int,default=4)
    p.add_argument('--val-every',type=int,default=50)
    p.add_argument('--val-samples',type=int,default=32)
    p.add_argument('--max-hours',type=float,default=1)
    p.add_argument('--stop-after',type=int,default=0,help='Stop at an optimizer boundary for resume testing')
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data'/'d3')
    p.add_argument('--init-adapter',type=Path)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--budget-stage',choices=['d3','d4','assessment'])
    a=p.parse_args()
    if min(a.steps,a.grad_accum,a.val_every,a.val_samples,a.patience)<1 or a.max_hours<=0:p.error('Positive limits required')
    if a.budget_stage:
        sys.path.insert(0,str(ROOT))
        from src.training.budget import SharedBudget
        with SharedBudget(a.budget_stage,a.max_hours*3600) as budget:return run(a,budget)
    return run(a,None)


def run(a,budget):
    sys.path.insert(0,str(ROOT))
    from src.training.loss import completion_loss
    from src.training.best_checkpoint import publish_best
    import torch,bitsandbytes as bnb
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA bf16 required')
    torch.set_num_threads(8);random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    folder=a.data_dir;base=ROOT/'data'/'models'/'Qwen3-4B-HF'
    config={'steps':a.steps,'grad_accum':a.grad_accum,'val_every':a.val_every,
        'val_samples':a.val_samples,'patience':a.patience,'seed':42,'lr':a.lr,'rank':8,
        'init_adapter_sha256':digest(a.init_adapter/'adapter_model.safetensors') if a.init_adapter else None,
        'seq_limit':2048,'train_sha256':digest(folder/'train_tokens.jsonl'),
        'val_sha256':digest(folder/'val_tokens.jsonl'),'model_manifest_sha256':digest(base/'download_manifest.json')}
    state=None
    if a.resume:
        state=torch.load(a.resume/'state.pt',map_location='cpu',weights_only=False)
        state['config'].setdefault('init_adapter_sha256',None)
        if {k:v for k,v in state['config'].items() if k!='validation_ids'}!=config:
            raise ValueError('Resume config/data/model mismatch')
        if not a.output.exists():raise ValueError('Resume must use its existing run directory')
        if a.resume.resolve().parent!=a.output.resolve():raise ValueError('Checkpoint belongs to another run')
    else:
        a.output.mkdir(parents=True,exist_ok=False)
    def read_rows(name):
        rows=[json.loads(l) for l in (folder/f'{name}_tokens.jsonl').open()]
        if not rows or any(len(r['input_ids'])>2048 for r in rows):raise ValueError('Invalid token data')
        return rows
    rows=read_rows('train');all_val=read_rows('val')
    # Fixed round-robin source balance for validation only.
    buckets={}
    for r in all_val:buckets.setdefault(r['source'],[]).append(r)
    rng=random.Random(42)
    for bucket in buckets.values():rng.shuffle(bucket)
    val=[]
    while len(val)<min(a.val_samples,len(all_val)):
        for key in sorted(buckets):
            if buckets[key] and len(val)<a.val_samples:val.append(buckets[key].pop())
    config['validation_ids']=[r['input_sha256'] for r in val]
    # Compare config after validation IDs have been reconstructed as well.
    if state and state['config']!=config:raise ValueError('Validation cohort mismatch')
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
        quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    for module in (model.get_input_embeddings(),model.get_output_embeddings()):module.to(dtype=torch.bfloat16)
    if state:model=PeftModel.from_pretrained(model,a.resume/'adapter',is_trainable=True,local_files_only=True)
    elif a.init_adapter:model=PeftModel.from_pretrained(model,a.init_adapter,is_trainable=True,local_files_only=True)
    else:model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',
        task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    parameters=[v for n,v in model.named_parameters() if v.requires_grad]
    assert all('lora_' in n for n,v in model.named_parameters() if v.requires_grad)
    optimizer=bnb.optim.AdamW8bit(parameters,lr=a.lr)
    warmup=max(1,math.ceil(a.steps*.1))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:
        (step+1)/warmup if step<warmup else max(0,(a.steps-step)/max(1,a.steps-warmup)))
    step=0;cursor=0;best=float('inf');bad=0;history=[];best_checkpoint=None
    if state:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        step=state['step'];cursor=state['cursor'];best=state['best'];bad=state['bad']
        history=state['history'];best_checkpoint=state['best_checkpoint']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    orders={}
    def next_row():
        nonlocal cursor
        epoch,offset=divmod(cursor,len(rows))
        if epoch not in orders:
            orders.clear();order=list(range(len(rows)));random.Random(42+epoch).shuffle(order);orders[epoch]=order
        row=rows[orders[epoch][offset]];cursor+=1;return row
    def loss_for(row):
        ids=torch.tensor([row['input_ids']],device='cuda')
        start=next(i for i,value in enumerate(row['labels']) if value!=-100)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return completion_loss(model,ids,torch.ones_like(ids),
                torch.tensor([row['labels']],device='cuda'),start)
    def evaluate():
        model.eval();total=0.;tokens=0;by_source={}
        with torch.no_grad():
            for row in val:
                count=sum(x!=-100 for x in row['labels']);loss=float(loss_for(row))
                if not math.isfinite(loss):raise ValueError('Nonfinite validation loss')
                total+=loss*count;tokens+=count
                entry=by_source.setdefault(row['source'],[0.,0]);entry[0]+=loss*count;entry[1]+=count
        model.train()
        torch.cuda.empty_cache()
        return {'loss':total/tokens,'source_loss':{k:v[0]/v[1] for k,v in by_source.items()},'examples':len(val)}
    def checkpoint(reason):
        nonlocal best_checkpoint
        path=a.output/f'checkpoint-{step:06d}'
        temporary=a.output/f'.checkpoint-{step:06d}.partial'
        temporary.mkdir(exist_ok=False)
        model.save_pretrained(temporary/'adapter',safe_serialization=True)
        torch.save({'config':config,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'step':step,'cursor':cursor,'best':best,'bad':bad,'history':history,
            'best_checkpoint':best_checkpoint,'python_rng':random.getstate(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temporary/'state.pt')
        (temporary/'metadata.json').write_text(json.dumps({'step':step,'reason':reason,'config':config},indent=2))
        temporary.rename(path)
        from src.training.budget import atomic_json
        atomic_json(a.output/'latest.json',{'checkpoint':path.name,'step':step})
        if best_checkpoint==path.name:publish_best(a.output,path,step,best)
        return path
    def status(reason):
        (a.output/'status.json').write_text(json.dumps({'status':reason,'step':step,'target_steps':a.steps,
            'last_checkpoint_step':last_saved,
            'session_elapsed_seconds':time.perf_counter()-start,
            'shared_session_seconds_remaining':budget.remaining() if budget else None,
            'examples_consumed':cursor,'best_validation_loss':best,'best_checkpoint':best_checkpoint,
            'best_adapter_manifest':'best.json',
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'config':config},indent=2))
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();model.train();last_saved=step
    if state and best_checkpoint:
        publish_best(a.output,a.output/best_checkpoint,int(best_checkpoint.rsplit('-',1)[1]),best)
        status('running')
    if not state:
        metrics=evaluate();best=metrics['loss'];history.append({'step':0,'validation':metrics})
        best_checkpoint='checkpoint-000000';checkpoint('baseline');status('running')
        print('Initial validation:',metrics,flush=True)
    try:
        while step<a.steps:
            if budget and budget.remaining()<180:
                if step>last_saved:checkpoint('shared_budget_stop');last_saved=step
                status('shared_budget_stop');break
            optimizer.zero_grad(set_to_none=True);losses=[]
            for micro in range(a.grad_accum):
                loss=loss_for(next_row())
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                losses.append(float(loss.detach()));(loss/a.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            del loss
            torch.cuda.empty_cache()
            history.append({'step':step,'train_loss':sum(losses)/len(losses),'lr':scheduler.get_last_lr()[0]})
            status('running')
            print(f'Step {step}/{a.steps}: loss={history[-1]["train_loss"]:.4f}',flush=True)
            stop=(a.stop_after and step>=a.stop_after) or time.perf_counter()-start>=a.max_hours*3600-180 or (budget and budget.remaining()<180)
            if step%a.val_every==0 or step==a.steps or stop:
                metrics=evaluate();history[-1]['validation']=metrics
                improvement=best-metrics['loss']
                if improvement>0:
                    best=metrics['loss'];best_checkpoint=f'checkpoint-{step:06d}'
                bad=0 if improvement>.001 else bad+1
                checkpoint('periodic');last_saved=step;status('running')
                (a.output/'history.json').write_text(json.dumps(history,indent=2))
                print('Validation:',metrics,flush=True)
                if bad>=a.patience:status('early_stopped');break
            if stop:status('bounded_stop');break
        else:status('completed')
    except BaseException as exc:
        # Resume from the last complete optimizer boundary, never a partial microbatch.
        status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed')
        raise
    print('Run state:',a.output/'status.json',flush=True)


if __name__=='__main__':main()
