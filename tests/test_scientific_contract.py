"""Check scientific failure cases and accounting, without requiring model weights."""
import math
import tempfile
import unittest
from pathlib import Path

from trustbench.io import assert_disjoint, read_jsonl, write_jsonl
from trustbench.metrics import (abstention_metrics, accept, binomial_tail, entropy_score,
                               exact_match, facttest_threshold, parse_value_choice,
                               repetition_metrics, selective_metrics)
from trustbench.runner import draw_seed, extract_factual


class ScientificContract(unittest.TestCase):
    def test_fpr_is_not_selective_error(self):
        rows = []
        for correct, accepted, n in [(True, True, 20), (True, False, 80), (False, True, 45), (False, False, 855)]:
            rows.extend(dict(status='ok', correct=correct, accepted=accepted) for _ in range(n))
        result = selective_metrics(rows)
        self.assertAlmostEqual(result['false_acceptance_given_incorrect'], 0.05)
        self.assertAlmostEqual(result['selective_error'], 45 / 65)
        self.assertEqual(result['correct_accepted_per_query'], 0.02)

    def test_failure_keeps_query_denominator(self):
        result = selective_metrics([dict(status='ok', correct=True, accepted=True), dict(status='error')])
        self.assertEqual(result['correct_accepted_per_query'], 0.5)
        self.assertEqual(result['failed_or_ungraded'], 1)

    def test_all_reject_accuracy_undefined(self):
        result = selective_metrics([dict(status='ok', correct=False, accepted=False)])
        self.assertIsNone(result['accepted_accuracy'])
        self.assertEqual(result['coverage_among_graded'], 0)

    def test_insufficient_calibration_rejects_all(self):
        policy = facttest_threshold([0.] * 89)
        self.assertEqual(policy['policy'], 'reject_all')
        self.assertFalse(accept(1000, policy))
        self.assertEqual(facttest_threshold([0.] * 90)['policy'], 'strictly_greater')

    def test_ties_do_not_break_false_acceptance_control(self):
        policy = facttest_threshold([-1.] * 100)
        self.assertFalse(accept(-1, policy))
        self.assertTrue(accept(0, policy))

    def test_binomial_tail_known_example(self):
        self.assertAlmostEqual(binomial_tail(2, 3, 0.5), 0.5)
        self.assertEqual(binomial_tail(4, 3, 0.5), 0)

    def test_order_statistic_meets_bound_and_is_minimal(self):
        policy = facttest_threshold(list(range(500)), alpha=.05, delta=.01)
        k = policy['order_statistic_one_based']
        self.assertLessEqual(binomial_tail(k, 500, .95), .01)
        self.assertGreater(binomial_tail(k - 1, 500, .95), .01)

    def test_entropy_uses_all_samples(self):
        self.assertEqual(entropy_score(['same'] * 5), 0)
        self.assertAlmostEqual(entropy_score(['a', 'b']), -math.log(2))

    def test_missing_period_protocols_are_distinct(self):
        self.assertEqual(extract_factual('Paris', 'preserve_missing_period'), 'Paris')
        self.assertEqual(extract_factual('Paris', 'released_period_slice'), 'Pari')

    def test_exact_matching_rejects_substring_collision(self):
        self.assertFalse(exact_match('Paris, Texas', ['Paris']))
        self.assertTrue(exact_match('The Hague.', ['Hague']))

    def test_abstention_reports_over_refusal(self):
        result = abstention_metrics([dict(status='ok', should_abstain=False, abstained=True),
                                     dict(status='ok', should_abstain=True, abstained=True),
                                     dict(status='ok', should_abstain=True, abstained=None)])
        self.assertEqual(result['over_abstention'], 1)
        self.assertEqual(result['abstention_precision'], .5)
        self.assertEqual(result['failed_or_ungraded'], 1)

    def test_option_swap_maps_semantic_action(self):
        self.assertEqual(parse_value_choice('I choose Option B; reason', {'A': 'second', 'B': 'first'}), 'first')
        self.assertIsNone(parse_value_choice('I choose Option A or I choose Option B', {'A': 'first', 'B': 'second'}))

    def test_invalid_draws_are_not_stability(self):
        rows = [dict(item_id='x', repetition=0, status='ok', choice='A'),
                dict(item_id='x', repetition=1, status='ok', choice=None)]
        result = repetition_metrics(rows, 2)
        self.assertIsNone(result['any_flip_rate_complete_only'])
        self.assertEqual(result['valid_choice_rate'], .5)

    def test_resume_seed_is_independent_of_processing_order(self):
        first = draw_seed(679, 'item', 3, 'candidate')
        draw_seed(679, 'other', 3, 'candidate')
        self.assertEqual(first, draw_seed(679, 'item', 3, 'candidate'))
        self.assertNotEqual(first, draw_seed(679, 'item', 3, 'sample-0'))

    def test_group_split_leakage_detected(self):
        with self.assertRaises(ValueError):
            assert_disjoint([dict(group_id='passage')], [dict(group_id='passage')])

    def test_records_round_trip_without_nan(self):
        scratch = Path(__file__).resolve().parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / 'records.jsonl'
            record = dict(item_id='é', risk=None)
            write_jsonl(path, [record])
            self.assertEqual(read_jsonl(path), [record])
            with self.assertRaises(ValueError):
                write_jsonl(path, [dict(risk=float('nan'))])


if __name__ == '__main__':
    unittest.main()
