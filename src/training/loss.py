"""Exact completion-only causal loss without unused prompt vocabulary logits."""
import torch.nn.functional as F


def completion_loss(model,input_ids,attention_mask,labels,completion_start):
    if input_ids.shape[0]!=1 or not 0<completion_start<input_ids.shape[1]:
        raise ValueError('Expected one example with a nonempty prompt and completion')
    # Last prompt position predicts the first answer token. The final answer
    # position predicts no supervised next token and is omitted from the loss.
    count=input_ids.shape[1]-completion_start
    logits=model(input_ids=input_ids,attention_mask=attention_mask,
                 logits_to_keep=count+1,use_cache=False).logits[:,:-1,:]
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                           labels[:,completion_start:].reshape(-1),ignore_index=-100)
