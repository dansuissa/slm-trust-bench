import copy
import importlib.util
from pathlib import Path
import unittest

from trustbench import review as module


class ReviewScoringTests(unittest.TestCase):
    def example(self):
        packet=[{'review_id':'1','context_and_question':'Context. Question?', 'dataset_should_abstain':False,
                 'reference_answers':['Paris'],'response':'Not enough context.','response_quality':'complete'}]
        annotations=copy.deepcopy(packet)
        annotations[0].update(reviewer='test fixture',reviewed_at='2026-10-23T14:00:00+00:00',
              response_state='explicit_abstention',abstained=True,correct=False,notes='Synthetic over-abstention fixture.')
        mapping={'1':{'run':'toy','item_id':'one','group_id':'passage'}}
        return packet,annotations,mapping

    def test_all_refusal_does_not_look_like_success_on_answerable_questions(self):
        records=module.validate_review(*self.example())
        metrics=module.summarize(records,bootstrap=0)['toy']
        self.assertEqual(metrics['over_abstention'],1.)
        self.assertEqual(metrics['answerable_accuracy_among_graded'],0.)
        self.assertIsNone(metrics['abstention_recall'])

    def test_pending_labels_are_not_scored_as_human_judgments(self):
        packet,annotations,mapping=self.example()
        annotations[0]['reviewer']=None
        with self.assertRaisesRegex(ValueError,'pending'):
            module.validate_review(packet,annotations,mapping)

    def test_altered_responses_and_contradictory_labels_are_rejected(self):
        packet,annotations,mapping=self.example()
        annotations[0]['response']='Paris'
        with self.assertRaisesRegex(ValueError,'source response'):
            module.validate_review(packet,annotations,mapping)
        packet,annotations,mapping=self.example()
        annotations[0]['abstained']=False
        with self.assertRaisesRegex(ValueError,'contradicts'):
            module.validate_review(packet,annotations,mapping)
