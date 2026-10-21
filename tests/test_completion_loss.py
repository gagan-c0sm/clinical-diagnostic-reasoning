import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from src.training.loss import completion_loss


class CompletionLossTests(unittest.TestCase):
    def test_matches_full_loss_and_gradients(self):
        torch.manual_seed(42)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,
            intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
            num_key_value_heads=2,head_dim=8,attention_dropout=0))
        ids=torch.tensor([[1,4,7,3,12,9,13,2]])
        labels=ids.clone();labels[:,:5]=-100
        full=model(input_ids=ids,labels=labels,use_cache=False).loss
        full.backward()
        reference={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad()
        compact=completion_loss(model,ids,torch.ones_like(ids),labels,5)
        compact.backward()
        torch.testing.assert_close(full,compact,rtol=1e-6,atol=1e-6)
        for n,p in model.named_parameters():
            if n in reference:torch.testing.assert_close(reference[n],p.grad,rtol=1e-5,atol=1e-7)


if __name__=='__main__':unittest.main()
