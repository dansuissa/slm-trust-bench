"""Prepare shuffled local review records without filling semantic judgments."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

from .io import completed_run_records, digest, read_jsonl, validate_items, write_json, write_jsonl


def prepare(runs, data, out, per_label=None):
    runs, data, out = Path(runs), Path(data), Path(out)
    source = {r['item_id']: r for r in validate_items(read_jsonl(data))}
    if any(row['task'] != 'abstention' or type(row['should_abstain']) is not bool for row in source.values()):
        raise ValueError('Context review requires labeled answerability data')
    selected = set(source)
    if per_label is not None:
        if per_label < 1:
            raise ValueError('Per-label sample size must be positive')
        selected = set()
        for label in [False, True]:
            eligible = sorted((r for r in source.values() if r['should_abstain'] is label),
                              key=lambda r: sha256(('review-679:' + r['item_id']).encode()).hexdigest())
            if len(eligible) < per_label:
                raise ValueError('Insufficient items in an answerability stratum')
            selected.update(row['item_id'] for row in eligible[:per_label])
    packet, mapping = [], {}
    for metadata in sorted(runs.glob('*/run.json')):
        folder = metadata.parent
        run = json.loads(metadata.read_text(encoding='utf-8'))
        if run['config']['task'] != 'abstention':
            raise ValueError('Review directory includes a non-context run')
        if run['fingerprint']['data'] != digest(data):
            raise ValueError('Run and review data hashes differ')
        records = completed_run_records(folder / 'responses.jsonl')
        if set(run['selected_items']) != set(source):
            raise ValueError('Run and source item sets differ')
        for record in records:
            if record['item_id'] not in selected:
                continue
            row = source[record['item_id']]
            identifier = sha256(f"679:{folder.name}:{row['item_id']}:{record['repetition']}".encode()).hexdigest()[:20]
            if identifier in mapping:
                raise ValueError('Review ID collision')
            packet.append({'review_id':identifier, 'context_and_question':row['messages'][-1]['content'],
                           'dataset_should_abstain':row['should_abstain'], 'reference_answers':row['references'],
                           'response':record['response'], 'response_quality':record['generation']['response_quality'],
                           'reviewer':None, 'response_state':None, 'abstained':None, 'correct':None,
                           'evidence':None, 'notes':None, 'reviewed_at':None})
            mapping[identifier] = {'run':folder.name, 'item_id':row['item_id'], 'group_id':row['group_id']}
    if not packet:
        raise ValueError('No completed context runs found')
    random.Random(679).shuffle(packet)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / 'unannotated_review.jsonl', packet)
    write_json(out / 'condition_mapping.json', mapping)
    write_json(out / 'provenance.json', {'data_sha256':digest(data), 'per_label_per_condition':per_label,
                                       'records':len(packet), 'semantic_labels_filled':0, 'seed':679})
    return {'review_records':len(packet), 'semantic_labels_filled':0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-label', type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.runs, args.data, args.out, args.per_label)))


if __name__ == '__main__':
    main()
