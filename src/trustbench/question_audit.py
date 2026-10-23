"""Prepare relation-aware development corrections and a reference-review ledger."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .io import digest, read_jsonl, write_json, write_jsonl
from .protocols import transform
from .repairs import repair_question


def ordered(rows, purpose):
    return sorted(rows, key=lambda r: sha256(f'cpu-plan-2026-10-23:{purpose}:{r["item_id"]}'.encode()).hexdigest())


def prepare(root, out):
    root, out = Path(root), Path(out)
    source = root / 'data/prepared/pararel_calibration_clean.jsonl'
    rows = ordered(read_jsonl(source), 'factual-development')
    partitions = {'development':rows[:100], 'calibration':ordered(rows[100:], 'factual-calibration')[:500]}
    corrected, ledger = {}, []
    for split, items in partitions.items():
        corrected[split] = []
        for item in items:
            revised = transform(item, 'factual_repaired', {})
            _, rule, _ = repair_question(item)
            if rule == 'unrecognized_affected_template':
                raise ValueError('Question-template coverage changed')
            corrected[split].append(revised)
            ledger.append({'split':split, 'item_id':item['item_id'], 'relation':item['relation'],
                           'original_question':item['messages'][-1]['content'],
                           'revised_question':revised['messages'][-1]['content'], 'rule':rule,
                           'reference_answers':deepcopy(item['references']), 'group_id':revised['group_id'],
                           'reference_adjudicated':False, 'reviewer':None, 'evidence':None})
    development = {r['group_id'] for r in corrected['development']}
    groups = defaultdict(list)
    for row in corrected['calibration']:
        groups[row['group_id']].append(row)
    usable, quarantine = [], []
    for group, values in groups.items():
        if group in development:
            reason = 'shared_with_development'
        elif len({tuple(r['references']) for r in values}) > 1:
            reason = 'different_reference_sets_pending_review'
        else:
            usable.append(values[0])
            values = values[1:]
            reason = 'duplicate_relation_subject'
        quarantine.extend({'item_id':r['item_id'], 'reason':reason} for r in values)
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / 'development.jsonl', corrected['development'])
    write_jsonl(out / 'calibration.jsonl', usable)
    write_jsonl(out / 'reference_review.jsonl', ledger)
    write_json(out / 'quarantine.json', quarantine)
    result = {'source_sha256':digest(source), 'development':len(corrected['development']),
              'calibration':len(usable), 'quarantined_calibration_records':len(quarantine),
              'repair_rules':dict(Counter(row['rule'] for row in ledger if row['rule'])),
              'reference_values_changed':False, 'reference_adjudication_complete':False,
              'test_data_used':False}
    write_json(out / 'audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.workspace, args.out), indent=2))


if __name__ == '__main__':
    main()
