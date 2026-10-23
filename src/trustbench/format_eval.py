"""Score a declared questionnaire format without assigning semantic quality labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .io import completed_run_records, digest, read_jsonl, write_json
from .metrics import parse_value_choice, repetition_metrics
from .repairs import parse_choice_only, parse_structured


def evaluate(run, data, contract):
    run, data = Path(run), Path(data)
    metadata = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    if metadata['config']['task'] != 'value' or metadata['fingerprint']['data'] != digest(data):
        raise ValueError('Run must be a value task with the supplied frozen data')
    items = {r['item_id']: r for r in read_jsonl(data)}
    records = completed_run_records(run / 'responses.jsonl')
    parsers = {'choice_only':parse_choice_only, 'neutral_structured':parse_structured}
    scored = []
    for row in records:
        if contract == 'released':
            choice = parse_value_choice(row['response'], items[row['item_id']]['option_map'])
            parsed = {'choice':choice, 'valid':choice is not None}
        else:
            parsed = parsers[contract](row['response'], items[row['item_id']]['option_map'])
        scored.append(dict(row, choice=parsed['choice'], parser_valid=parsed['valid']))
    return {
        'contract':contract, 'draws':len(scored), 'successful_parses':sum(r['parser_valid'] for r in scored),
        'candidate_quality':dict(Counter(r['generation']['response_quality'] for r in scored)),
        'choice_repetition_diagnostic':repetition_metrics(scored, metadata['config']['repetitions']),
        'data_sha256':digest(data), 'responses_sha256':digest(run / 'responses.jsonl'),
        'semantic_quality':None,
        'interpretation':'Syntax and choice repetition only. Contracts differ in required content; parse success is not reasoning quality.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--contract', choices=['released','neutral_structured','choice_only'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Analysis output exists; choose a new path')
    write_json(args.out, evaluate(args.run, args.data, args.contract))


if __name__ == '__main__':
    main()
