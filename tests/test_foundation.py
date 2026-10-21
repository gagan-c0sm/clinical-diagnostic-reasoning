import json
import unittest
from pathlib import Path
from src.data import DiReCTDataset
from src.data.reconcile import grouped_split
from src.data.structures import InputContent, KGNode, KnowledgeGraph
from src.pipeline.pipeline import DiagnosticPipeline
from src.pipeline.schema import validate_tree, to_nested
from src.pipeline.client import LocalClient

ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
    def generate(self, *args):
        return next(self.outputs)


class FoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = DiReCTDataset(ROOT/'mimic-iv-ext-direct-1.0.0')

    def test_identity_and_group_isolation(self):
        samples = self.dataset.samples
        self.assertEqual(len({s.sample_id for s in samples}), 511)
        splits = grouped_split(samples)
        patients = {k:{s.patient_id for s in v} for k,v in splits.items()}
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            self.assertFalse(patients[a] & patients[b])
        for name, items in splits.items():
            saved = json.loads((ROOT/'data'/'splits_v2'/f'{name}.json').read_text())
            self.assertEqual([s.sample_id for s in items], [r['sample_id'] for r in saved])
        self.assertEqual(sum(map(len,splits.values())), 511)
        with self.assertRaises(ValueError):
            self.dataset.get_sample('11389640-DS-16')

    def test_training_export_excludes_dev_test(self):
        folder = ROOT/'data'/'splits_v2'
        train = {r['sample_id'] for r in json.loads((folder/'train.json').read_text())}
        records = [json.loads(l) for l in (folder/'direct_train.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(records), 335)
        self.assertTrue({r['sample_id'] for r in records} <= train)
        for record in records:
            s = self.dataset.get_sample(record['sample_id'])
            self.assertFalse(validate_tree(json.loads(record['output'])))

    def test_rejects_hallucination_and_builds_valid_tree(self):
        kg = KnowledgeGraph('Example', KGNode('Suspected', [KGNode('Confirmed')]),
                            {'Suspected':'Evidence', 'Confirmed':'Evidence'})
        client = FakeClient([{'category':'Example'},
            {'selected_ids':[999, 1]},
            {'choice':'Suspected','evidence_ids':['o1'],'rationale':'Supports suspicion'},
            {'choice':'Confirmed','evidence_ids':['o1'],'rationale':'Supports confirmation'}])
        result = DiagnosticPipeline(client, {'Example':kg}, ['Example']).run(InputContent(input1='finding'))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['observations']), 1)
        self.assertIn('input1:rejected_invalid_passage_id', result['warnings'])
        self.assertFalse(validate_tree(result['nodes'], InputContent(input1='finding')))
        self.assertTrue(next(iter(to_nested(result['nodes']))).startswith('Confirmed$'))

    def test_missing_kg_and_stop(self):
        result = DiagnosticPipeline(FakeClient([{'category':'Gastritis'}]), {}, ['Gastritis']).run(InputContent(input1='text'))
        self.assertEqual(result['status'], 'unsupported_kg')
        self.assertIsNone(result['diagnosis'])
        kg = KnowledgeGraph('Example', KGNode('Root'), {})
        result = DiagnosticPipeline(FakeClient([{'category':'Example'},
            {'selected_ids':[1]},
            {'choice':'STOP','evidence_ids':[],'rationale':'insufficient'}]),
            {'Example':kg}, ['Example']).run(InputContent(input1='text'))
        self.assertEqual(result['status'], 'abstained')
        self.assertFalse(result['nodes'])

    def test_disconnected_cycle_rejected(self):
        nodes = {'a':{'content':'a','type':'Intermedia','parent':'b','children':['b']},
                 'b':{'content':'b','type':'Intermedia','parent':'a','children':['a']}}
        self.assertIn('unreachable_nodes', validate_tree(nodes))

    def test_truncated_generation_rejected(self):
        client = LocalClient()
        client.request = lambda *args: {'choices':[{'finish_reason':'length',
            'message':{'content':'{"category":"Example"}'}}]}
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            client.generate('classify','test',{})
        self.assertEqual(client.calls[0]['finish_reason'], 'length')

    def test_shortlist_can_correct_first_category(self):
        client = FakeClient([
            {'evidence_summary':'Synthetic findings','ranked_categories':['Wrong','Example']},
            {'assessments':[{'category':'Wrong','assessment':'Unsupported'},
                            {'category':'Example','assessment':'Supported'}], 'category':'Example'},
            {'selected_ids':[1]},
            {'choice':'Root','evidence_ids':['o1'],'rationale':'Finding supports criterion'}])
        kg = KnowledgeGraph('Example', KGNode('Root'), {'Root':'finding'})
        result = DiagnosticPipeline(client, {'Example':kg}, ['Wrong','Example'],
                                    routing='shortlist').run(InputContent(input1='finding'))
        self.assertEqual(result['initial_category'], 'Wrong')
        self.assertEqual(result['category'], 'Example')
        self.assertEqual(result['status'], 'complete')

    def test_empty_shortlist_abstains(self):
        result = DiagnosticPipeline(FakeClient([{'evidence_summary':'No evidence',
            'ranked_categories':[]}]), {}, ['Example'], routing='shortlist').run(InputContent(input1='text'))
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(result['category'], 'Uncertain')


if __name__ == '__main__':
    unittest.main()
