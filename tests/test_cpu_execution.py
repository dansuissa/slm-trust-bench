"""CPU recovery, rendering and numerical likelihood tests without loading a model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trustbench.cpu import journal_records, likelihood_summary, messages_for, run


class CPUExecutionTests(unittest.TestCase):
    def test_resume_reuses_each_completed_draw_and_rejects_configuration_change(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        calls = []

        class FakeGenerator:
            parameters, defaults, load_seconds = 10, {}, 0.

            def __init__(self, config, monitor):
                pass

            def generate(self, messages, settings, seed, likelihood=False):
                calls.append(seed)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return {'text': 'Paris', 'response_quality': 'complete', 'end_to_end_seconds': 0., 'seed': seed}

        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory).resolve()
            item = {'item_id': 'one', 'group_id': 'one', 'dataset': 'toy', 'task': 'factual',
                    'split': 'development', 'messages': [{'role': 'user', 'content': 'Capital of France?'}],
                    'references': ['Paris'], 'should_abstain': None}
            data, config_path, output = root / 'data.jsonl', root / 'config.json', root / 'run'
            data.write_text(json.dumps(item) + '\n', encoding='utf-8')
            config = {'cpu_threads': 1, 'dtype': 'float32', 'prompt_rendering': 'native_chat',
                      'question_instruction': 'Short answer.', 'data': str(data), 'task': 'factual',
                      'repetitions': 1, 'score_samples': 2, 'seed': 679, 'candidate': {},
                      'score_decoding': {}, 'purpose': 'recovery test'}
            config_path.write_text(json.dumps(config), encoding='utf-8')
            with patch('trustbench.cpu.CPUGenerator', FakeGenerator), patch('trustbench.cpu.environment', return_value={}), patch('builtins.print'):
                with self.assertRaises(KeyboardInterrupt):
                    run(config_path, output)
                first = journal_records(output / 'draws.jsonl')[0]
                run(config_path, output, resume=True)
                self.assertEqual(len(calls), 4)
                self.assertEqual(journal_records(output / 'draws.jsonl')[0], first)
                self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'generation_complete')
                config['seed'] = 680
                config_path.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                    run(config_path, output, resume=True)

    def test_partial_last_draw_is_preserved_and_recovered(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / 'draws.jsonl'
            path.write_bytes(b'{"draw": 1}\n{"dra')
            with self.assertRaises(ValueError):
                journal_records(path)
            self.assertEqual(journal_records(path, recover=True), [{'draw': 1}])
            backups = list(path.parent.glob('draws.jsonl.before_recovery.*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b'{"draw": 1}\n{"dra')

    def test_corruption_in_the_middle_is_not_silently_repaired(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / 'draws.jsonl'
            path.write_bytes(b'{"draw":1}\nwrong\n{"draw":2}\n')
            with self.assertRaises(ValueError):
                journal_records(path, recover=True)

    def test_raw_and_native_conditions_receive_identical_question_content(self):
        item = {'task': 'factual', 'messages': [{'role': 'user', 'content': 'A question?'}]}
        native = messages_for(item, {'question_instruction': 'Short answer.', 'prompt_rendering': 'native_chat'})
        raw = messages_for(item, {'question_instruction': 'Short answer.', 'prompt_rendering': 'raw_content'})
        self.assertEqual(native, raw)

    def test_raw_likelihood_excludes_eos_and_does_not_use_sampling_scores(self):
        import torch
        logits = [torch.tensor([[0., 0., 0.]]), torch.tensor([[0., 0., 0.]])]
        result = likelihood_summary(logits, [1, 2], {2})
        self.assertAlmostEqual(result['answer_log_likelihood_sum'], -1.098612288, places=6)
        self.assertAlmostEqual(result['answer_log_likelihood_mean'], -1.098612288, places=6)
        self.assertEqual(result['answer_likelihood_token_count'], 1)
        self.assertIsNone(likelihood_summary([logits[0]], [2], {2})['answer_log_likelihood_mean'])
        with self.assertRaises(ValueError):
            likelihood_summary(logits, [1], {2})


if __name__ == '__main__':
    unittest.main()
