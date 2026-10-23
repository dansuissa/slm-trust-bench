"""CPU-only calibration implementation diagnostics with known score distributions.

This exercises a published guarantee and an explicit shift counterexample. It is
neither a new theorem nor a model experiment. Full runs belong on a remote CPU.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

from .io import write_json
from .metrics import binomial_tail, facttest_threshold
from .stats import wilson_interval


def score(rng, distribution):
    if distribution == 'continuous_uniform':
        return rng.random()
    if distribution == 'discrete_five_levels':
        return rng.randrange(5) / 4
    if distribution == 'high_confidence_atom':
        return 0.0 if rng.random() < .7 else -rng.random()
    raise ValueError(distribution)


def population_false_acceptance(threshold, distribution):
    if threshold is None:
        return 0.0
    if distribution == 'continuous_uniform':
        return 1 - threshold
    if distribution == 'discrete_five_levels':
        return sum(value > threshold for value in [0., .25, .5, .75, 1.]) / 5
    if distribution == 'high_confidence_atom':
        return 0.0 if threshold >= 0 else .7 + .3 * (-threshold)
    raise ValueError(distribution)


def diagnostics(trials, seed):
    cells = []
    for n in [0, 10, 89, 90, 200, 500]:
        policy = facttest_threshold([0.] * n)
        k = policy['order_statistic_one_based']
        for distribution in ['continuous_uniform', 'discrete_five_levels', 'high_confidence_atom']:
            rng = random.Random(f'{seed}:{n}:{distribution}')
            violations = 0
            reject_all = 0
            rates = []
            shifted_violations = 0
            for _ in range(trials):
                ordered = sorted(score(rng, distribution) for _ in range(n))
                threshold = ordered[k - 1] if k <= n else None
                fpr = population_false_acceptance(threshold, distribution)
                rates.append(fpr)
                reject_all += threshold is None or (distribution == 'high_confidence_atom' and threshold == 0)
                violations += fpr > .05
                if distribution == 'continuous_uniform':
                    shifted_fpr = 0 if threshold is None else 1 - threshold ** 10
                    shifted_violations += shifted_fpr > .05
            cells.append({'n_incorrect_calibration': n, 'distribution': distribution, 'trials': trials,
                          'alpha': .05, 'delta': .01, 'order_statistic': k,
                          'theoretical_violation_upper_bound': binomial_tail(k, n, .95),
                          'empirical_violation_fraction': violations / trials,
                          'violation_fraction_wilson_95': wilson_interval(violations, trials),
                          'mean_population_false_acceptance': sum(rates) / trials,
                          'reject_all_policy_or_top_atom_threshold_fraction': reject_all / trials,
                          'shift_to_beta10_violation_fraction': shifted_violations / trials if distribution == 'continuous_uniform' else None})
    return {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
            'kind': 'synthetic implementation diagnostic; not model results or a proof',
            'warning': 'A small quick run cannot estimate a rare 1% event precisely. Shift has no asserted guarantee.', 'cells': cells}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=679)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    result = diagnostics(args.trials, args.seed)
    write_json(args.out, result)
    print(json.dumps({'output': str(args.out), 'cells': len(result['cells']), 'trials_per_cell': args.trials}))
