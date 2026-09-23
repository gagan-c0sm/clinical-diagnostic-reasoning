"""Offline mean-pooled biomedical embeddings using the existing torch stack."""
import hashlib,json
from pathlib import Path


class BiomedicalEncoder:
    def __init__(self,folder,device='cpu',batch_size=8):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;self.device=device;self.batch_size=batch_size;self.cache={}
        self.folder=Path(folder)
        self.manifest=json.loads((self.folder/'download_manifest.json').read_text())
        # Refuse a modified or incomplete model rather than fetching a replacement.
        for name,digest in self.manifest['sha256'].items():
            with (self.folder/name).open('rb') as source:
                if hashlib.file_digest(source,'sha256').hexdigest()!=digest:
                    raise ValueError(f'Evaluation model hash mismatch: {name}')
        self.tokenizer=AutoTokenizer.from_pretrained(self.folder,local_files_only=True,trust_remote_code=False)
        self.model=AutoModel.from_pretrained(self.folder,local_files_only=True,
            trust_remote_code=False,weights_only=True).to(device).eval()
        self.max_length=json.loads((self.folder/'sentence_bert_config.json').read_text())['max_seq_length']
        self.chunked_texts=0

    def prepare(self,texts):
        pending=list(dict.fromkeys(t for t in texts if t not in self.cache))
        torch=self.torch
        for start in range(0,len(pending),self.batch_size):
            batch=pending[start:start+self.batch_size]
            tokens=self.tokenizer(batch,padding=True,truncation=True,max_length=self.max_length,
                return_overflowing_tokens=True,return_tensors='pt')
            mapping=tokens.pop('overflow_to_sample_mapping')
            self.chunked_texts+=sum((mapping==i).sum().item()>1 for i in range(len(batch)))
            sums=torch.zeros(len(batch),self.model.config.hidden_size)
            weights=torch.zeros(len(batch),1)
            # Process a bounded number of chunks at once, even for long paragraphs.
            for offset in range(0,len(mapping),self.batch_size):
                part={k:v[offset:offset+self.batch_size].to(self.device) for k,v in tokens.items()}
                with torch.inference_mode():
                    hidden=self.model(**part).last_hidden_state
                    mask=part['attention_mask'].unsqueeze(-1)
                    vectors=(hidden*mask).sum(1).cpu()
                    counts=mask.sum(1).cpu()
                sums.index_add_(0,mapping[offset:offset+self.batch_size],vectors)
                weights.index_add_(0,mapping[offset:offset+self.batch_size],counts.float())
            vectors=torch.nn.functional.normalize(sums/weights.clamp_min(1),dim=1)
            self.cache.update(zip(batch,vectors))

    def similarity(self,left,right):
        self.prepare([left,right])
        return max(-1.,min(1.,float(self.cache[left]@self.cache[right])))
