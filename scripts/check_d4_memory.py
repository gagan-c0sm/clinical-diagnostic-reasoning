"""Replay the two examples from the failed update without saving updates."""
import json,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
from src.training.budget import SharedBudget,atomic_json


def main():
    import torch,bitsandbytes as bnb
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training,PeftModel
    from src.training.loss import completion_loss
    with SharedBudget('assessment',300):
        torch.set_num_threads(8)
        base=ROOT/'data'/'models'/'Qwen3-4B-HF'
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
        model.config.use_cache=False
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        for module in (model.get_input_embeddings(),model.get_output_embeddings()):module.to(dtype=torch.bfloat16)
        checkpoint=ROOT/'data'/'training'/'d4_session_600'/'checkpoint-000050'
        model=PeftModel.from_pretrained(model,checkpoint/'adapter',is_trainable=True,local_files_only=True)
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=bnb.optim.AdamW8bit(parameters,lr=5e-5)
        state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        rows=[json.loads(l) for l in (ROOT/'data'/'d4_compact'/'train_tokens.jsonl').open()]
        order=list(range(len(rows)));random.Random(42).shuffle(order)
        model.train();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();losses=[]
        for i in [168,169]:
            row=rows[order[i]];start=next(j for j,x in enumerate(row['labels']) if x!=-100)
            ids=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor([row['labels']],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=completion_loss(model,ids,torch.ones_like(ids),labels,start)
            (loss/2).backward();losses.append(float(loss.detach()));del loss
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
        report={'status':'passed','replayed_failed_update':85,'losses':losses,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
            'note':'Temporary memory test only; weights discarded. Durable checkpoint50 unchanged.'}
        atomic_json(ROOT/'data'/'training'/'d4_memory_repair.json',report);print(report,flush=True)


if __name__=='__main__':main()
