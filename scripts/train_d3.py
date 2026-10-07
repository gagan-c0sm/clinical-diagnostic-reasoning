"""Project-local NF4 QLoRA training. Default: two optimizer-step smoke trial."""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'data'/'.hf_cache')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--seq-len',type=int,default=1024)
    parser.add_argument('--grad-accum',type=int,default=1)
    parser.add_argument('--output',type=Path,default=ROOT/'data'/'training'/'d3_trial_1024')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data'/'d3')
    args=parser.parse_args()
    if min(args.steps,args.seq_len,args.grad_accum)<1:parser.error('Positive values required')
    if args.output.exists():raise FileExistsError('Use a new output directory')
    import torch
    import bitsandbytes as bnb
    import transformers,peft
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel,get_peft_model_state_dict
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    args.output.mkdir(parents=True)
    report={'config':vars(args)|{'output':str(args.output),'data_dir':str(args.data_dir)},'versions':{
        'torch':torch.__version__,'transformers':transformers.__version__,
        'peft':peft.__version__,'bitsandbytes':bnb.__version__},
        'gpu':torch.cuda.get_device_name(), 'losses':[],'status':'started'}
    report['data_manifest_sha256']=hashlib.sha256((args.data_dir/'manifest.json').read_bytes()).hexdigest()
    report['model_manifest_sha256']=hashlib.sha256((ROOT/'data'/'models'/'Qwen3-4B-HF'/'download_manifest.json').read_bytes()).hexdigest()
    def save_report():
        (args.output/'trial_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save_report()
    try:
        rows=[]
        for line in (args.data_dir/'train_tokens.jsonl').open():
            row=json.loads(line)
            if len(row['input_ids'])<=args.seq_len:rows.append(row)
        if len(rows)<2:raise ValueError('Too few complete examples fit')
        # Stress the requested context size rather than validating tiny samples.
        if args.steps<=2:rows.sort(key=lambda r:len(r['input_ids']),reverse=True)
        else:random.shuffle(rows)
        report['eligible_training_examples']=len(rows)
        report['trial_sequence_lengths']=[len(r['input_ids']) for r in rows[:args.steps*args.grad_accum]]
        base=ROOT/'data'/'models'/'Qwen3-4B-HF'
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
        def load_base():
            return AutoModelForCausalLM.from_pretrained(base,local_files_only=True,
                quantization_config=quant,device_map={'':0},torch_dtype=dtype,
                attn_implementation='sdpa')
        model=load_base();model.config.use_cache=False
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        # k-bit preparation upcasts frozen embeddings/head; retain bf16 storage.
        for module in (model.get_input_embeddings(),model.get_output_embeddings()):
            module.to(dtype=dtype)
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            bias='none',task_type='CAUSAL_LM'))
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert trainable and all('lora_' in n for n,p in trainable)
        report['trainable_parameters']=sum(p.numel() for n,p in trainable)
        optimizer=bnb.optim.AdamW8bit([p for n,p in trainable],lr=1e-4)
        def batch(row):
            ids=torch.tensor([row['input_ids']],device='cuda')
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
                    'labels':torch.tensor([row['labels']],device='cuda')}
        def loss_for(row):
            with torch.autocast('cuda',dtype=dtype):return model(**batch(row)).loss
        model.eval()
        with torch.no_grad():reference_before=float(loss_for(rows[0]))
        report['reference_loss_before']=reference_before
        model.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for micro in range(args.grad_accum):
                row=rows[(step*args.grad_accum+micro)%len(rows)]
                loss=loss_for(row)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                losses.append(float(loss.detach()));(loss/args.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_([p for n,p in trainable],1.0)
            optimizer.step()
            report['losses'].append(sum(losses)/len(losses))
            print(f'Step {step+1}: loss={report["losses"][-1]:.4f}',flush=True)
        torch.cuda.synchronize()
        report['training_seconds']=time.perf_counter()-start
        report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
        optimizer.zero_grad(set_to_none=True);model.eval()
        with torch.no_grad():reference_after=float(loss_for(rows[0]))
        report['reference_loss_after']=reference_after
        adapter_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        adapter=args.output/'adapter';model.save_pretrained(adapter,safe_serialization=True)
        AutoTokenizer.from_pretrained(base,local_files_only=True).save_pretrained(adapter)
        del optimizer,trainable,model,module,loss;gc.collect();torch.cuda.empty_cache()
        model=load_base();model.config.use_cache=False
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        for module in (model.get_input_embeddings(),model.get_output_embeddings()):module.to(dtype=dtype)
        model=PeftModel.from_pretrained(model,adapter,local_files_only=True)
        loaded_state=get_peft_model_state_dict(model)
        report['adapter_tensors_equal']=all(torch.equal(v,loaded_state[k].detach().cpu()) for k,v in adapter_state.items())
        if not report['adapter_tensors_equal']:raise ValueError('Saved adapter tensor mismatch')
        model.eval()
        with torch.no_grad():reloaded=float(loss_for(rows[0]))
        report['reloaded_reference_loss']=reloaded
        if not math.isfinite(reloaded) or abs(reloaded-reference_after)>.05:
            raise ValueError('Reloaded adapter loss differs beyond tolerance')
        report['status']='passed';save_report()
        print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if torch.cuda.is_available():report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        save_report();raise


if __name__=='__main__':main()
