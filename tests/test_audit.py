from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from trustbench.audit import audit_run
from trustbench.io import digest, write_json, write_jsonl
from trustbench.runner import draw_seed


class JournalAuditTests(unittest.TestCase):
    def test_complete_journal_passes_and_seed_or_candidate_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'run'
            item = {'item_id':'one', 'group_id':'passage', 'task':'abstention', 'dataset':'toy',
                    'split':'development', 'messages':[{'role':'user','content':'A question?'}],
                    'references':['An answer'], 'should_abstain':False}
            write_jsonl(root / 'data.jsonl', [item])
            generation = {'text':'An answer', 'seed':draw_seed(679, 'one', 0, 'candidate'),
                          'generation_seconds':1., 'end_to_end_seconds':1.1,
                          'input_tokens':10, 'output_tokens':2, 'response_quality':'complete'}
            event = {'item_id':'one', 'repetition':0, 'phase':'candidate', 'generation':generation}
            record = {'item_id':'one', 'repetition':0, 'group_id':'passage', 'status':'ok',
                      'should_abstain':False, 'response':'An answer', 'generation':generation}
            config = {'task':'abstention', 'data':'data.jsonl', 'seed':679, 'repetitions':1,
                      'max_context_tokens':100, 'model':{'id':'toy','revision':'0'*40}}
            write_json(folder / 'run.json', {'status':'generation_complete', 'config':config,
                       'fingerprint':{'data':digest(root / 'data.jsonl'), 'config':'fixture', 'code':{}},
                       'selected_items':['one'], 'parameter_count_loaded':1, 'environment':{},
                       'started_at_utc':'2026-10-23T00:00:00+00:00', 'finished_at_utc':'2026-10-23T00:00:02+00:00'})
            write_json(folder / 'summary.json', {'draws':1, 'responses':1, 'peak_rss_bytes_observed':100})
            write_jsonl(folder / 'draws.jsonl', [event])
            write_jsonl(folder / 'responses.jsonl', [record])
            self.assertEqual(audit_run(folder, root)['generation_calls'], 1)
            changed = deepcopy(event)
            changed['generation']['seed'] += 1
            write_jsonl(folder / 'draws.jsonl', [changed])
            with self.assertRaisesRegex(ValueError, 'Seed mismatch'):
                audit_run(folder, root)
            write_jsonl(folder / 'draws.jsonl', [event])
            record['response'] = 'A different answer'
            write_jsonl(folder / 'responses.jsonl', [record])
            with self.assertRaisesRegex(ValueError, 'Response text differs'):
                audit_run(folder, root)
