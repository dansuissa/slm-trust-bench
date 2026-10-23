from pathlib import Path
import tempfile
import unittest

from trustbench.io import digest, read_jsonl, write_json, write_jsonl
from trustbench.prepare_review import prepare


class ReviewPreparationTests(unittest.TestCase):
    def test_packets_are_paired_shuffled_and_unannotated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [{'item_id':str(i), 'group_id':f'passage-{i}', 'task':'abstention',
                      'dataset':'toy', 'split':'development', 'messages':[{'role':'user','content':f'Question {i}'}],
                      'references':['Answer'] if i % 2 else [], 'should_abstain':i % 2 == 0} for i in range(4)]
            data = root / 'data.jsonl'
            write_jsonl(data, items)
            for model in ['model-a', 'model-b']:
                folder = root / 'runs' / model
                write_json(folder / 'run.json', {'status':'generation_complete', 'config':{'task':'abstention','repetitions':1},
                                               'selected_items':[r['item_id'] for r in items], 'fingerprint':{'data':digest(data)}})
                write_jsonl(folder / 'responses.jsonl', [{'item_id':r['item_id'], 'repetition':0,
                            'response':'A response', 'generation':{'response_quality':'complete'}} for r in items])
            result = prepare(root / 'runs', data, root / 'review', per_label=1)
            self.assertEqual(result['review_records'], 4)
            packet = read_jsonl(root / 'review/unannotated_review.jsonl')
            self.assertTrue(all(r['reviewer'] is None and r['correct'] is None for r in packet))
            self.assertTrue(all('model' not in r['review_id'] for r in packet))
            self.assertEqual(sum(r['dataset_should_abstain'] for r in packet), 2)
            with self.assertRaises(FileExistsError):
                prepare(root / 'runs', data, root / 'review', per_label=1)
