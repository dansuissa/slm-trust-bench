"""Validate real review labels and score context answerability by model condition."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

from .metrics import abstention_metrics
from .stats import cluster_bootstrap


def validate_review(packet,annotations,mapping):
    expected={row['review_id']:row for row in packet}
    if len(expected)!=len(packet) or set(expected)!=set(mapping):
        raise ValueError('Review packet and mapping IDs must be unique and agree')
    actual={row['review_id']:row for row in annotations}
    if len(actual)!=len(annotations) or set(actual)!=set(expected):
        raise ValueError('Missing, duplicate, or unexpected annotation IDs')
    records=defaultdict(list)
    fixed=['context_and_question','dataset_should_abstain','reference_answers','response','response_quality']
    pending=[]
    for identifier,row in actual.items():
        if any(row.get(key)!=expected[identifier][key] for key in fixed):
            raise ValueError('An annotation changed a source response or question')
        if not row.get('reviewer') or not row.get('reviewed_at') or not row.get('response_state'):
            pending.append(identifier)
            continue
        if not isinstance(row['reviewer'],str) or not row['reviewer'].strip():
            raise ValueError('A reviewer identifier must be a nonempty string')
        if any(row.get(field) is not None and not isinstance(row[field],str) for field in ['evidence','notes']):
            raise ValueError('Evidence and notes must be text or null')
        timestamp=datetime.fromisoformat(row['reviewed_at'])
        if timestamp.tzinfo is None:
            raise ValueError('Review timestamps must include a timezone')
        state=row['response_state']
        if state not in {'answer','explicit_abstention','mixed','empty','unclear'}:
            raise ValueError('Unknown response state')
        abstained=True if state=='explicit_abstention' else False if state=='answer' else None
        if row.get('abstained') is not abstained:
            raise ValueError('Abstention label contradicts the reviewed response state')
        correct=row.get('correct')
        if correct is not None and type(correct) is not bool:
            raise ValueError('Correctness labels must be booleans or null')
        if row['dataset_should_abstain'] and correct is not None:
            raise ValueError('Correctness on unanswerable questions is outside this rubric')
        if not row['dataset_should_abstain'] and state in {'explicit_abstention','empty'} and correct is not False:
            raise ValueError('Abstention or empty output on an answerable question is incorrect')
        if not row.get('evidence') and not row.get('notes'):
            raise ValueError('Each completed review needs evidence or a short grading note')
        key=mapping[identifier]
        records[key['run']].append({'item_id':key['item_id'],'group_id':key['group_id'],
                'status':'ok','response_state':state,'abstained':abstained,
                'should_abstain':row['dataset_should_abstain'],'correct':correct})
    if pending:
        raise ValueError(f'{len(pending)} review records are pending; final semantic metrics were not computed')
    return records


def summarize(records,bootstrap=2000):
    result={}
    for condition,rows in records.items():
        metric=abstention_metrics(rows)
        answerable=[row for row in rows if row['should_abstain'] is False]
        graded=[row for row in answerable if type(row['correct']) is bool]
        metric.update(answerable_queries=len(answerable),answerable_graded=len(graded),
                      answerable_correct=sum(row['correct'] for row in graded),
                      answerable_accuracy_among_graded=sum(row['correct'] for row in graded)/len(graded) if graded else None,
                      unique_passage_groups=len({row['group_id'] for row in rows}),
                      response_states={state:sum(row['response_state']==state for row in rows) for state in ['answer','explicit_abstention','mixed','empty','unclear']})
        if bootstrap:
            metric['abstention_recall_interval']=cluster_bootstrap(rows,lambda sample:abstention_metrics(sample)['abstention_recall'],bootstrap)
        result[condition]=metric
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap<0: parser.error('Bootstrap count must be nonnegative')
    packet=[json.loads(line) for line in args.packet.read_text(encoding='utf-8').splitlines()]
    annotated=[json.loads(line) for line in args.annotations.read_text(encoding='utf-8').splitlines()]
    mapping=json.loads(args.mapping.read_text(encoding='utf-8'))
    records=validate_review(packet,annotated,mapping)
    report={'scored_at_utc':datetime.now(timezone.utc).isoformat(),'annotation_records':len(annotated),
            'input_sha256':{name:sha256(path.read_bytes()).hexdigest() for name,path in
                            [('packet',args.packet),('annotations',args.annotations),('mapping',args.mapping)]},
            'scoring_tool_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'bootstrap_resamples':args.bootstrap,'bootstrap_seed':679,
            'reviewers':sorted({row['reviewer'] for row in annotated}),'conditions':summarize(records,args.bootstrap),
            'interpretation':'Context answerability on balanced development data. Annotation validation cannot establish that a reviewer judgment is correct; independently inspect the evidence.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
