import unittest
from src.demo.case_io import parse_case,render_report


class CaseIOTests(unittest.TestCase):
    def test_rejects_answer_fields_and_invalid_input(self):
        for data in [{},{'input1':None},{'input1':'text','diagnosis':'answer'},[]]:
            with self.assertRaises(ValueError):parse_case(data)
        case=parse_case({'input1':'Symptom','case_id':'test'})
        self.assertEqual(case.input1,'Symptom');self.assertEqual(case.input6,'')

    def test_report_escapes_patient_and_model_text(self):
        result={'status':'abstained','category':'<script>alert(1)</script>', 'diagnosis':None,'nodes':{},'warnings':[]}
        report=render_report(result,parse_case({'input1':'<img src=x onerror=alert(1)>'}))
        self.assertNotIn('<script>',report);self.assertNotIn('<img',report)
        self.assertIn('&lt;img',report)


if __name__=='__main__':unittest.main()
