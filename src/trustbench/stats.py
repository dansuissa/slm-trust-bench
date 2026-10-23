"""Intervals over base questions or passages; repeated draws stay in their group."""
from collections import defaultdict
import math
import random


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError('Invalid binomial counts')
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def percentile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def cluster_bootstrap(records, statistic, repetitions=1000, seed=679):
    if repetitions < 1:
        raise ValueError('Positive bootstrap repetitions required')
    groups = defaultdict(list)
    for record in records:
        groups[record['group_id']].append(record)
    keys = sorted(groups)
    if len(keys) < 2:
        return {'interval': None, 'reason': 'fewer than two independent groups', 'groups': len(keys)}
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sample = [row for _ in keys for row in groups[rng.choice(keys)]]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    return {'interval': [percentile(values, .025), percentile(values, .975)] if values else None,
            'groups': len(keys), 'requested_resamples': repetitions, 'defined_resamples': len(values),
            'method': 'cluster percentile bootstrap; conditional on recorded calibration policy', 'seed': seed}
