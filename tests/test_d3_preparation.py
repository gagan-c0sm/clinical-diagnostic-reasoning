import json
import unittest
from pathlib import Path
from scripts.prepare_d3_data import fingerprint
from scripts.profile_d3_tokens import encode

ROOT=Path(__file__).resolve().parents[1]


class D3Tests(unittest.TestCase):
    def test_normalization_and_frozen_split_isolation(self):
        self.assertEqual(fingerprint('Example\n QUESTION'),fingerprint('example question'))
        groups=[]
        for split in ['train','val','test']:
            groups.append({json.loads(l)['input_sha256'] for l in
                (ROOT/'data'/'d3'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()})
        for a,b in [(0,1),(0,2),(1,2)]:self.assertFalse(groups[a]&groups[b])

    def test_completion_only_loss_mask(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'data'/'models'/'Qwen3-4B-HF',local_files_only=True)
        row={'instruction':'Analyze evidence.','input':'Synthetic case.','output':'Candidate diagnosis.'}
        ids,labels=encode(row,tokenizer)
        self.assertEqual(len(ids),len(labels))
        prefix=labels.index(next(x for x in labels if x!=-100))
        self.assertTrue(all(x==-100 for x in labels[:prefix]))
        self.assertEqual(tokenizer.decode(labels[prefix:]),row['output']+'<|im_end|>')


if __name__=='__main__':unittest.main()
