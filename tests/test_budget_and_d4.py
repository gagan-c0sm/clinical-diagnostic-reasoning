import json,random,tempfile,unittest
from pathlib import Path
from src.training.budget import SharedBudget,initialize,atomic_json

ROOT=Path(__file__).resolve().parents[1]


class BudgetAndD4Tests(unittest.TestCase):
    def test_shared_budget_lock_and_cumulative_charge(self):
        test_root=ROOT/'data'/'test_tmp';test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(test_root.resolve()))
            path=Path(tmp)/'budget.json'
            for _ in range(2):
                with SharedBudget('d4',600,path) as budget:
                    self.assertGreater(budget.remaining(),0)
                    with self.assertRaises(RuntimeError):
                        with SharedBudget('d3',600,path):pass
            ledger=json.loads(path.read_text())
            self.assertEqual(len(ledger['sessions']),2)
            self.assertIsNone(ledger['active'])
            self.assertGreater(sum(s['seconds'] for s in ledger['sessions']),0)
            ledger['charged_historical_seconds']=14400;atomic_json(path,ledger)
            with self.assertRaises(RuntimeError):
                with SharedBudget('d4',600,path):pass
            self.assertFalse(path.with_suffix('.lock').exists())

    def test_d4_partition_and_targets(self):
        all_ids={}
        for split,original in [('train','train'),('val','dev')]:
            allowed={r['sample_id'] for r in json.loads((ROOT/'data'/'splits_v2'/f'{original}.json').read_text())}
            rows=[json.loads(l) for l in (ROOT/'data'/'d4_compact'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            tokens=[json.loads(l) for l in (ROOT/'data'/'d4_compact'/f'{split}_tokens.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),len(tokens));all_ids[split]={r['sample_id'] for r in rows}
            self.assertTrue(all_ids[split]<=allowed)
            for r,t in zip(rows,tokens):
                self.assertLessEqual(len(t['input_ids']),2048)
                self.assertEqual(len(t['labels']),len(t['input_ids']))
                answer=json.loads(r['output'])
                if r['task_type']=='perception':self.assertTrue(answer['selected_ids'])
                if r['task_type']=='reason':
                    self.assertTrue(answer['evidence_ids']);self.assertNotEqual(answer['choice'],'STOP')
                    self.assertNotIn('Setup node',answer['rationale'])
            if split=='train':self.assertEqual(len({r['category'] for r in rows}),25)
        self.assertFalse(all_ids['train']&all_ids['val'])

    def test_curated_data_excludes_pilot_and_validation(self):
        folder=ROOT/'data'/'d3'
        tokens=[json.loads(l) for l in (folder/'train_tokens.jsonl').read_text().splitlines()]
        order=list(range(len(tokens)));random.Random(42).shuffle(order)
        pilot={tokens[i]['input_sha256'] for i in order[:800]}
        curated={json.loads(l)['input_sha256'] for l in (ROOT/'data'/'d3_curated'/'train.jsonl').read_text(encoding='utf-8').splitlines()}
        val={json.loads(l)['input_sha256'] for l in (folder/'val.jsonl').read_text(encoding='utf-8').splitlines()}
        self.assertEqual(len(curated),1600);self.assertFalse(curated&pilot);self.assertFalse(curated&val)


if __name__=='__main__':unittest.main()
