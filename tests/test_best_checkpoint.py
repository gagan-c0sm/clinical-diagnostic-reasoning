import json,tempfile,unittest
from pathlib import Path
from src.training.best_checkpoint import publish_best

ROOT=Path(__file__).resolve().parents[1]


class BestCheckpointTests(unittest.TestCase):
    def test_best_is_saved_only_for_strictly_lower_validation_loss(self):
        test_root=ROOT/'data'/'test_tmp';test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as tmp:
            run=Path(tmp).resolve()
            self.assertTrue(run.is_relative_to(test_root.resolve()))
            def checkpoint(step,content):
                p=run/f'checkpoint-{step:06d}';(p/'adapter').mkdir(parents=True)
                (p/'adapter'/'adapter_model.safetensors').write_bytes(content)
                (p/'adapter'/'adapter_config.json').write_text('{}')
                (p/'state.pt').write_bytes(b'resume')
                return p
            original=checkpoint(50,b'first')
            first=publish_best(run,original,50,.9)
            self.assertEqual((run/first['adapter']/'adapter_model.safetensors').read_bytes(),b'first')
            worse=checkpoint(100,b'worse')
            self.assertEqual(publish_best(run,worse,100,1.1)['step'],50)
            self.assertEqual(publish_best(run,worse,100,.9)['step'],50)
            better=checkpoint(150,b'better')
            updated=publish_best(run,better,150,.89999)
            self.assertEqual(updated['step'],150)
            self.assertEqual(json.loads((run/'best.json').read_text())['validation_loss'],.89999)
            self.assertTrue((run/first['adapter']).exists())
            self.assertEqual((original/'adapter'/'adapter_model.safetensors').read_bytes(),b'first')
            with self.assertRaises(ValueError):publish_best(run,better,150,float('nan'))


if __name__=='__main__':unittest.main()
