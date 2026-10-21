import copy,json,unittest
from src.demo.education import explain,validate_explanation,step_material
from src.demo.library import catalog,case_for,saved_for,records
from src.demo.case_io import parse_case

class EducationTests(unittest.TestCase):
    def test_library_is_development_only_and_inputs_have_no_answers(self):
        from src.demo.runner import ROOT
        allowed=json.loads((ROOT/'data/splits_v2/dev.json').read_text(encoding='utf-8'))
        self.assertEqual(len(catalog()),len(allowed))
        self.assertEqual(sum(c['replay'] for c in catalog()),25)
        for c in catalog():
            case=case_for(c['id']);parse_case(case)
            self.assertEqual(set(case),{'case_id',*[f'input{i}' for i in range(1,7)]})
            if c['replay']:
                result=saved_for(c['id'])
                self.assertNotIn('expected_diagnosis',result)
                self.assertNotIn('expected_category',result)

    def test_education_cannot_mutate_diagnosis_and_rejects_fabricated_citations(self):
        result=next(saved_for(key) for key,r in records().items() if r['replay'] and r['saved']['nodes'])
        original=copy.deepcopy(result);step=step_material(result)[0]
        text={k:'Explanation' for k in ('plain','why','unresolved','definition')}
        raw=dict(text,checks=[],terms=[])
        class Fake:
            def generate(self,*args):return raw
        generated=explain(result,Fake())
        self.assertEqual(result,original);self.assertEqual(generated['mode'],'generated')
        bad=dict(raw,checks=[{'criterion_quote':'INVENTED CRITERION','status':'supporting','evidence_ids':['nonexistent'],'explanation':'fake'}])
        with self.assertRaises(ValueError):validate_explanation(bad,step)
        bad=dict(raw,terms=[{'term':'term not present in material','meaning':'invented'}])
        with self.assertRaises(ValueError):validate_explanation(bad,step)
        class Broken:
            def generate(self,*args):raise ValueError('incomplete generation')
        fallback=explain(result,Broken())
        self.assertTrue(fallback['errors']);self.assertEqual(fallback['steps'][0]['mode'],'source_only')

if __name__=='__main__':unittest.main()
