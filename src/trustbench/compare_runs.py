"""Compare deterministic replay content while excluding runtime measurements."""
import argparse
import json
import math
from pathlib import Path

from .io import digest, read_jsonl, write_json


def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    metadata = [json.loads((p / 'run.json').read_text(encoding='utf-8')) for p in [reference, candidate]]
    if any(r['status'] != 'generation_complete' for r in metadata):
        raise ValueError('Both runs must be complete')
    if metadata[0]['config'] != metadata[1]['config'] or metadata[0]['fingerprint']['data'] != metadata[1]['fingerprint']['data']:
        raise ValueError('Replay data or configuration differs')
    def draws(path):
        rows = read_jsonl(path / 'draws.jsonl')
        result = {(r['item_id'], r['repetition'], r['phase']):r['generation'] for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate draw identity')
        return result
    left, right = draws(reference), draws(candidate)
    if set(left) != set(right):
        raise ValueError('Replay draw identities differ')
    fields = ['text', 'input_tokens', 'output_tokens', 'seed', 'rendered_prompt_sha256',
              'response_quality', 'finish_reason', 'answer_likelihood_token_count', 'likelihood_eos_policy']
    mismatches = []
    maximum_error = 0.
    for key in left:
        for name in fields:
            if left[key].get(name) != right[key].get(name):
                mismatches.append({'item_id':key[0], 'repetition':key[1], 'phase':key[2], 'field':name})
        a, b = left[key].get('raw_token_log_probabilities', []), right[key].get('raw_token_log_probabilities', [])
        if len(a) != len(b):
            mismatches.append({'item_id':key[0], 'repetition':key[1], 'phase':key[2], 'field':'likelihood_token_count'})
        else:
            for x, y in zip(a, b):
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError('Non-finite replay likelihood')
                maximum_error = max(maximum_error, abs(x-y))
    return {'draws_compared':len(left), 'content_mismatches':mismatches,
            'maximum_absolute_token_log_probability_difference':maximum_error,
            'matched_with_absolute_tolerance_1e_6':not mismatches and maximum_error <= 1e-6,
            'reference_draws_sha256':digest(reference / 'draws.jsonl'),
            'candidate_draws_sha256':digest(candidate / 'draws.jsonl'),
            'reference_code_sha256':metadata[0]['fingerprint']['code'],
            'candidate_code_sha256':metadata[1]['fingerprint']['code']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    write_json(args.out, result)
    if not result['matched_with_absolute_tolerance_1e_6']:
        raise SystemExit('Replay differs; inspect the comparison artifact')


if __name__ == '__main__':
    main()
