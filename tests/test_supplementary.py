import unittest
from src.data.download import format_medcase_for_sft, format_medical_o1_for_sft


class FormatterTests(unittest.TestCase):
    def test_official_medical_o1_capitalized_fields(self):
        result = format_medical_o1_for_sft({"Question":"A synthetic case?",
            "Complex_CoT":"Evidence supports a candidate.", "Response":"Candidate diagnosis."})
        self.assertEqual(result['input'], 'A synthetic case?')
        self.assertIn('Evidence supports', result['output'])
        self.assertIn('Candidate diagnosis.', result['output'])

    def test_medcase_missing_fields_and_shared_wrapper(self):
        self.assertIsNone(format_medcase_for_sft({'case_prompt':None}))
        result = format_medcase_for_sft({'case_prompt':'Synthetic case',
            'diagnostic_reasoning':'Evidence', 'final_diagnosis':'Candidate'})
        self.assertTrue(all(result.get(k) for k in ['instruction','input','output']))


if __name__ == '__main__':
    unittest.main()
