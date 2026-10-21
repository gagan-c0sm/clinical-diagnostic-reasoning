import unittest
from pathlib import Path
from src.data.loader import ClinicalSample
from src.data.structures import InputContent,RecordNode,NodeType
from src.evaluation.metrics import matching,prf,score_case


class ExactEncoder:
    def prepare(self,texts):pass
    def similarity(self,a,b):return float(a==b)


class EvaluationTests(unittest.TestCase):
    def test_maximum_matching_does_not_reuse_reference(self):
        self.assertEqual(len(matching([[.95,.85],[.9,-1]],.8)),2)
        self.assertEqual(len(matching([[1],[1]],.8)),1)
        self.assertEqual(prf(0,0,1)['recall'],0.)

    def test_source_and_diagnosis_gates_and_abstention(self):
        sample=ClinicalSample('x','Example','Confirmed',Path('Example/x.json'),InputContent(input1='Finding',input2='Finding'))
        sample.record_nodes={
            'd':RecordNode('d','Confirmed',NodeType.INTERMEDIA,'d',None,['c']),
            'c':RecordNode('c','Supports diagnosis',NodeType.CAUSE,'c','d',['i']),
            'i':RecordNode('i','Finding',NodeType.INPUT,'i','c',[],1)}
        sample.root_node_id='d'
        nodes={'d':{'content':'Confirmed','type':'Intermedia','parent':None,'children':['c']},
            'c':{'content':'Supports diagnosis','type':'Cause','parent':'d','children':['i']},
            'i':{'content':'Finding','type':'Input','parent':'c','children':[],'source_field':1}}
        row={'category':'Example','diagnosis':'Confirmed','status':'complete',
            'observations':[{'text':'Finding','source_field':1}],'nodes':nodes}
        good=score_case(row,sample,ExactEncoder())
        self.assertEqual(good['explanation_steps']['f1'],1.)
        self.assertEqual(good['observations_semantic']['f1'],1.)
        sample.pdd='Broad metadata category'
        audited=score_case(row,sample,ExactEncoder())
        self.assertFalse(audited['diagnosis_correct'])
        self.assertTrue(audited['diagnosis_annotation_root_correct'])
        row['observations'][0]['source_field']=2
        self.assertEqual(score_case(row,sample,ExactEncoder())['observations_semantic']['matched'],0)
        nodes['d']['content']='Different diagnosis'
        self.assertEqual(score_case(row,sample,ExactEncoder())['explanation_steps']['matched'],0)
        empty=score_case({'status':'abstained','nodes':{},'observations':[]},sample,ExactEncoder())
        self.assertEqual(empty['explanation_steps']['reference'],1)
        self.assertEqual(empty['explanation_steps']['recall'],0.)


if __name__=='__main__':unittest.main()
