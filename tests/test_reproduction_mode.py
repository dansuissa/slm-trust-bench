"""Original-data reconstruction must be explicit and cannot silently bypass benchmark guards."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trustbench.cli import main
from trustbench.io import write_json, write_jsonl


class ReproductionModeTests(unittest.TestCase):
    def test_shared_original_records_require_explicit_mode(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            calibration = root / 'calibration'
            records = [dict(item_id=f'cal:{i}', group_id='shared_question', repetition=0,
                            status='ok', correct=False, score=float(i)) for i in range(90)]
            config = dict(task='factual', repetitions=1, data='calibration.jsonl', score_samples=5)
            write_jsonl(calibration / 'responses.jsonl', records)
            write_json(calibration / 'run.json', dict(status='generation_complete', config=config,
                                                     selected_items=[r['item_id'] for r in records]))
            args = ['trustbench', 'calibrate', '--records', str(calibration / 'responses.jsonl'),
                    '--out', str(calibration / 'policy.json')]
            with patch('sys.argv', args), self.assertRaisesRegex(ValueError, 'independent record'):
                main()
            with patch('sys.argv', args + ['--published-reproduction']), redirect_stdout(io.StringIO()):
                main()
            policy = json.loads((calibration / 'policy.json').read_text(encoding='utf-8'))
            self.assertTrue(policy['published_reproduction'])
            self.assertFalse(policy['independence_assumption_verified'])
            test = root / 'test'
            write_jsonl(test / 'responses.jsonl', [dict(item_id='test:0', group_id='shared_question', repetition=0,
                                                       status='ok', correct=False, score=100.)])
            write_json(test / 'run.json', dict(status='generation_complete', config=dict(config, data='test.jsonl'),
                                               selected_items=['test:0']))
            args = ['trustbench', 'evaluate', '--task', 'factual', '--records', str(test / 'responses.jsonl'),
                    '--calibration', str(calibration / 'policy.json'), '--out', str(test / 'metrics.json')]
            with patch('sys.argv', args), self.assertRaisesRegex(ValueError, 'Reproduction mode differs'):
                main()
            with patch('sys.argv', args + ['--published-reproduction']), redirect_stdout(io.StringIO()):
                main()
            result = json.loads((test / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(result['calibration_test_question_overlap'], 1)
            self.assertTrue(result['published_reproduction'])


if __name__ == '__main__':
    unittest.main()
