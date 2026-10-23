"""Preparation, validation, local smoke checks, and explicit GPU-stage commands."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .data import prepare
from .io import assert_disjoint, completed_run_records, digest, read_jsonl, validate_items, write_json, write_jsonl
from .metrics import accept, abstention_metrics, facttest_threshold, repetition_metrics, selective_metrics

ROOT = Path.cwd().resolve()


def main():
    parser = argparse.ArgumentParser(prog='trustbench')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    commands.add_parser('validate')
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--out', type=Path, required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--config', type=Path, required=True)
    run_parser.add_argument('--out', type=Path, required=True)
    run_parser.add_argument('--resume', action='store_true')
    judge_parser = commands.add_parser('judge')
    judge_parser.add_argument('--config', type=Path, required=True)
    judge_parser.add_argument('--run', type=Path, required=True)
    calibration = commands.add_parser('calibrate')
    calibration.add_argument('--records', type=Path, required=True)
    calibration.add_argument('--out', type=Path, required=True)
    calibration.add_argument('--alpha', type=float, default=0.05)
    calibration.add_argument('--delta', type=float, default=0.01)
    calibration.add_argument('--published-reproduction', action='store_true',
                             help='Preserve repeated published question records; mark assumptions as unverified')
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--records', type=Path, required=True)
    evaluate.add_argument('--task', choices=['factual', 'abstention', 'value'], required=True)
    evaluate.add_argument('--calibration', type=Path)
    evaluate.add_argument('--repetitions', type=int, default=10)
    evaluate.add_argument('--bootstrap', type=int, default=0)
    evaluate.add_argument('--published-reproduction', action='store_true')
    evaluate.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(ROOT)
    elif args.command == 'validate':
        audit = json.loads((ROOT / 'data/prepared/audit.json').read_text(encoding='utf-8'))
        for name, summary in audit['datasets'].items():
            if digest(ROOT / 'data/prepared' / name) != summary['sha256']:
                raise ValueError(f'Prepared data changed: {name}')
            validate_items(read_jsonl(ROOT / 'data/prepared' / name))
        squad = read_jsonl(ROOT / 'data/prepared/squad2.jsonl')
        assert_disjoint(*[[x for x in squad if x['research_split'] == split]
                          for split in ['development', 'calibration', 'test']])
        assert_disjoint(read_jsonl(ROOT / 'data/prepared/pararel_calibration_clean.jsonl'),
                        read_jsonl(ROOT / 'data/prepared/pararel_test_clean.jsonl'))
        result = {'status': 'passed', 'prepared_hashes_and_schemas': True, 'squad_passage_splits_disjoint': True,
                  'pararel_clean_question_splits_disjoint': True,
                  'pararel_published_overlap': audit['pararel_question_group_overlap']}
    elif args.command == 'smoke':
        # Synthetic fixtures exercise accounting and persistence without pretending to be model output.
        records = [dict(item_id=f'synthetic:{i}', group_id=f'synthetic:{i}', status='ok', correct=correct,
                        accepted=accepted) for i, (correct, accepted) in enumerate([(True, True), (False, True),
                                                                                 (True, False), (False, False)])]
        records.append(dict(item_id='synthetic:4', group_id='synthetic:4', status='error'))
        write_jsonl(args.out / 'synthetic_records.jsonl', records)
        result = {'data_kind': 'synthetic; no model executed', 'metrics': selective_metrics(read_jsonl(args.out / 'synthetic_records.jsonl')),
                  'reject_all_calibration': facttest_threshold([-1.0] * 10)}
        write_json(args.out / 'smoke.json', result)
    elif args.command == 'run':
        from .runner import run
        result = run(ROOT, args.config, args.out, args.resume)
    elif args.command == 'judge':
        from .runner import judge
        result = judge(ROOT, args.config, args.run)
    elif args.command == 'calibrate':
        records = completed_run_records(args.records)
        if any(r['status'] != 'ok' or type(r.get('correct')) is not bool for r in records):
            raise ValueError('Calibration contains failed or ungraded candidates; resolve before calibrating')
        if len({r['group_id'] for r in records}) != len(records) and not args.published_reproduction:
            raise ValueError('Calibration requires one independent record per group')
        result = facttest_threshold([r['score'] for r in records if not r['correct']], args.alpha, args.delta)
        result['calibration_groups'] = [r['group_id'] for r in records]
        result['published_reproduction'] = args.published_reproduction
        result['independence_assumption_verified'] = False
        result['source_sha256'] = digest(args.records)
        metadata = args.records.parent / 'run.json'
        if not metadata.exists():
            raise ValueError('Calibration requires run metadata for protocol verification')
        config = json.loads(metadata.read_text(encoding='utf-8'))['config']
        result['protocol'] = {k: v for k, v in config.items() if k not in {'data', 'max_items', 'research_split'}}
        write_json(args.out, result)
    else:
        records = deepcopy(completed_run_records(args.records))
        if args.task == 'factual':
            if not args.calibration:
                raise ValueError('Factual evaluation requires a calibration artifact')
            policy = json.loads(args.calibration.read_text(encoding='utf-8'))
            overlap = {r['group_id'] for r in records} & set(policy['calibration_groups'])
            if args.published_reproduction != policy.get('published_reproduction', False):
                raise ValueError('Reproduction mode differs between calibration and evaluation')
            if overlap and not args.published_reproduction:
                raise ValueError('Calibration/test group leakage')
            config = json.loads((args.records.parent / 'run.json').read_text(encoding='utf-8'))['config']
            protocol = {k: v for k, v in config.items() if k not in {'data', 'max_items', 'research_split'}}
            if protocol != policy['protocol']:
                raise ValueError('Calibration and evaluation protocols differ')
            for record in records:
                if record['status'] == 'ok':
                    record['accepted'] = accept(record['score'], policy)
            result = selective_metrics(records)
            result.update(published_reproduction=args.published_reproduction,
                          calibration_test_question_overlap=len(overlap),
                          guarantee_assumptions='require independent sampling and a frozen applicable protocol; not verified by software')
        elif args.task == 'abstention':
            result = abstention_metrics(records)
            answerable = [r for r in records if r.get('should_abstain') is False]
            graded = [r for r in answerable if r['status'] == 'ok' and type(r.get('correct')) is bool]
            result.update(answerable_queries=len(answerable), answerable_graded=len(graded),
                          answerable_correct=sum(r['correct'] for r in graded),
                          answerable_accuracy_among_graded=sum(r['correct'] for r in graded) / len(graded) if graded else None)
        else:
            result = repetition_metrics(records, args.repetitions)
        if args.bootstrap:
            from .stats import cluster_bootstrap
            if args.task == 'factual':
                result['selective_error_interval'] = cluster_bootstrap(records,
                    lambda rows: selective_metrics(rows)['selective_error'], args.bootstrap)
                result['correct_accepted_per_query_interval'] = cluster_bootstrap(records,
                    lambda rows: selective_metrics(rows)['correct_accepted_per_query'], args.bootstrap)
            elif args.task == 'abstention':
                result['abstention_recall_interval'] = cluster_bootstrap(records,
                    lambda rows: abstention_metrics(rows)['abstention_recall'], args.bootstrap)
            else:
                raise ValueError('Value repetition intervals require explicit item/variant resampling; not implemented')
        write_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
