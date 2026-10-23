import unittest

from trustbench.stats import cluster_bootstrap, wilson_interval


class IntervalTests(unittest.TestCase):
    def test_zero_observed_errors_does_not_mean_zero_uncertainty(self):
        lower, upper = wilson_interval(0, 10)
        self.assertAlmostEqual(lower, 0)
        self.assertGreater(upper, .25)
        self.assertIsNone(wilson_interval(0, 0))

    def test_repeated_draws_do_not_manufacture_independent_groups(self):
        rows = [dict(group_id='one_question', outcome=i % 2) for i in range(100)]
        result = cluster_bootstrap(rows, lambda sample: sum(r['outcome'] for r in sample) / len(sample))
        self.assertIsNone(result['interval'])
        self.assertEqual(result['groups'], 1)


if __name__ == '__main__':
    unittest.main()
