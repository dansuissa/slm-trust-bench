"""Verify completed CPU journals and summarize measurements without semantic grading."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .io import completed_run_records, digest, read_jsonl, validate_items, write_json
from .metrics import entropy_score, exact_match
from .runner import draw_seed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_run(folder, workspace, code_root=None):
    folder, workspace = Path(folder), Path(workspace)
    meta = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    require(meta['status'] == 'generation_complete', f'Incomplete run: {folder.name}')
    config = meta['config']
    source = workspace / config['data']
    require(digest(source) == meta['fingerprint']['data'], f'Data changed: {folder.name}')
    if code_root is not None:
        for name, expected in meta['fingerprint']['code'].items():
            require(digest(Path(code_root) / name) == expected, f'Code changed: {name}')
    items = {row['item_id']: row for row in validate_items(read_jsonl(source))}
    require(set(items) == set(meta['selected_items']), 'Selected items do not match input')
    records = completed_run_records(folder / 'responses.jsonl')
    draws = read_jsonl(folder / 'draws.jsonl')
    phases = ['candidate'] + ([f'sample-{i}' for i in range(config['score_samples'])]
                              if config['task'] == 'factual' else [])
    expected = {(identifier, repetition, phase) for identifier in items
                for repetition in range(config['repetitions']) for phase in phases}
    saved = {(row['item_id'], row['repetition'], row['phase']): row['generation'] for row in draws}
    require(len(saved) == len(draws) and set(saved) == expected, 'Missing or duplicate draws')
    for (identifier, repetition, phase), generation in saved.items():
        require(generation['seed'] == draw_seed(config['seed'], identifier, repetition, phase), 'Seed mismatch')
        require(isinstance(generation['text'], str), 'Non-text response')
        for name in ['generation_seconds', 'end_to_end_seconds']:
            require(math.isfinite(generation[name]) and generation[name] >= 0, 'Invalid timing')
        require(generation['output_tokens'] > 0 and generation['input_tokens'] > 0, 'Invalid token count')
        require(generation['input_tokens'] + generation['output_tokens'] <= config['max_context_tokens'], 'Context exceeded')
    for record in records:
        key = (record['item_id'], record['repetition'], 'candidate')
        require(record['generation'] == saved[key], 'Candidate differs from saved draw')
        require(record['response'] == saved[key]['text'], 'Response text differs from draw')
        item = items[record['item_id']]
        require(record['group_id'] == item['group_id'], 'Group changed')
        require(record['should_abstain'] is item['should_abstain'], 'Answerability changed')
        if config['task'] == 'factual':
            samples = [saved[(record['item_id'], record['repetition'], phase)] for phase in phases[1:]]
            require(record['samples'] == samples, 'Scoring samples differ from journal')
            require(record['correct'] == exact_match(record['response'], item['references']), 'String-match drift')
            require(record['score_raw_entropy'] == entropy_score([s['text'].strip() for s in samples]), 'Entropy drift')
            valid = bool(record['response'].strip()) and all(s['text'].strip() for s in samples)
            expected_score = record['score_raw_entropy'] if valid else -math.log(len(samples)) - 1.
            require(record['confidence_inputs_nonempty'] is bool(valid), 'Empty-input validity drift')
            require(record['score'] == expected_score, 'Stored confidence score drift')
            generation = record['generation']
            values = generation['raw_token_log_probabilities']
            require(len(values) == generation['output_tokens'] and all(math.isfinite(x) and x <= 1e-6 for x in values), 'Invalid token likelihoods')
            n = generation['answer_likelihood_token_count']
            require(n in {len(values), len(values) - 1}, 'Unexpected EOS accounting')
            require(n > 0, 'No answer likelihood tokens')
            require(math.isclose(sum(values[:n]), generation['answer_log_likelihood_sum'], abs_tol=1e-6), 'Likelihood sum differs')
            require(math.isclose(sum(values[:n]) / n, generation['answer_log_likelihood_mean'], abs_tol=1e-6), 'Likelihood mean differs')
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    require(summary['draws'] == len(draws) and summary['responses'] == len(records), 'Summary count mismatch')
    response_counts = Counter(record['response'].strip() for record in records)
    return {
        'run': folder.name, 'task': config['task'], 'model': config['model'],
        'items': len(items), 'responses': len(records), 'generation_calls': len(draws),
        'candidate_quality': dict(Counter(r['generation']['response_quality'] for r in records)),
        'empty_candidates': sum(not r['response'].strip() for r in records),
        'distinct_candidate_texts': len(response_counts),
        'largest_identical_text_count': max(response_counts.values(), default=0),
        'answerability_counts': dict(Counter(str(item['should_abstain']) for item in items.values())),
        'generation_timer_seconds': sum(d['generation']['end_to_end_seconds'] for d in draws),
        'peak_process_rss_bytes': summary['peak_rss_bytes_observed'],
        'parameter_count': meta['parameter_count_loaded'],
        'config_sha256': meta['fingerprint']['config'], 'data_sha256': digest(source),
        'code_sha256': meta['fingerprint']['code'],
        'draws_sha256': digest(folder / 'draws.jsonl'), 'responses_sha256': digest(folder / 'responses.jsonl'),
        'run_metadata_sha256': digest(folder / 'run.json'),
        'environment': meta['environment'], 'config': config,
        'started_at_utc': meta['started_at_utc'], 'finished_at_utc': meta['finished_at_utc'],
        'semantic_metrics': None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, action='append', required=True)
    parser.add_argument('--workspace', type=Path, required=True, help='Root containing the data paths recorded by each run')
    parser.add_argument('--code-root', type=Path, help='Original source directory, for historical code hash verification')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    folders = [p.parent for root in args.runs for p in sorted(root.glob('*/run.json'))]
    require(bool(folders), 'No run directories found')
    require(len({p.resolve() for p in folders}) == len(folders), 'Overlapping snapshot paths supplied')
    runs = [audit_run(folder, args.workspace, args.code_root) for folder in folders]
    require(len({row['run'] for row in runs}) == len(runs), 'Duplicate run names; do not pool historical snapshots')
    write_json(args.out, {
        'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'runs': runs,
        'completed_conditions': len(runs), 'generation_calls': sum(row['generation_calls'] for row in runs),
        'candidate_responses': sum(row['responses'] for row in runs),
        'source_code_hashes_verified': args.code_root is not None,
        'semantic_grading_performed_by_audit': False,
        'interpretation': 'Development measurements. Calls and repeated outputs are not independent observations. Semantic correctness and abstention have not been adjudicated.',
    })
    print(json.dumps({'conditions': len(runs), 'generation_calls': sum(r['generation_calls'] for r in runs)}))


if __name__ == '__main__':
    main()
