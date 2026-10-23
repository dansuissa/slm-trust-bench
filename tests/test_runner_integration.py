"""Exercise the actual run/cache path using an explicitly synthetic generator."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from trustbench.io import completed_run_records, read_jsonl, write_json, write_jsonl
from trustbench.runner import run


class SyntheticGenerator:
    parameters = 0
    generation_defaults = {'synthetic': True}
    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError))

    def __init__(self, config):
        pass

    def generate(self, messages, settings, seed, plain=False):
        return dict(text='Paris.', input_tokens=1, output_tokens=1, seconds=0,
                    seed=seed, peak_cuda_bytes=None, rendered_prompt_sha256='synthetic')


class RunnerIntegration(unittest.TestCase):
    def test_checkpointed_resume_and_protocol_change(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            data = [dict(item_id='synthetic:q', group_id='synthetic:q', dataset='synthetic', task='factual',
                         split='test', messages=[dict(role='user', content='Synthetic question')],
                         references=['Paris'], should_abstain=None)]
            write_jsonl(root / 'items.jsonl', data)
            config = dict(data='items.jsonl', task='factual', seed=679, repetitions=1, max_items=None,
                          candidate_prompt='Question: {question} Answer:', score_prompt='{question}',
                          candidate={'max_new_tokens': 15}, score_decoding={'max_new_tokens': 15},
                          score_samples=5, answer_extraction='preserve_missing_period', answer_rule='substring')
            path = root / 'config.json'
            write_json(path, config)
            with patch('trustbench.runner.HFGenerator', SyntheticGenerator), patch('trustbench.runner.importlib.metadata.version', return_value='synthetic'):
                result = run(root, path, root / 'run')
                self.assertEqual(result['responses'], 1)
                record = read_jsonl(root / 'run/responses.jsonl')[0]
                self.assertEqual(len(record['samples']), 5)
                self.assertTrue(record['correct'])
                self.assertEqual(len(completed_run_records(root / 'run/responses.jsonl')), 1)
                self.assertEqual(run(root, path, root / 'run', resume=True)['responses'], 1)
                config['score_samples'] = 10
                write_json(path, config)
                with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                    run(root, path, root / 'run', resume=True)

    def test_partial_runs_cannot_be_scored_as_complete(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            write_json(root / 'run.json', dict(status='running', selected_items=['q'], config={'repetitions': 1}))
            write_jsonl(root / 'responses.jsonl', [])
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                completed_run_records(root / 'responses.jsonl')


if __name__ == '__main__':
    unittest.main()
