"""Auditable input repairs and format parsers, separate from frozen inference code."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

VERSION='measurement-repair-v1'


def normalized_label(text):
    return ' '.join(text.casefold().split()).strip('\"\'')


SUBJECT_PATTERNS={
 'P101':[r'What field do(?:es)? (.+) work in\?'],
 'P103':[r'What is the native language of (.+)\?'],
 'P106':[r'What does (.+) work as\?'],
 'P108':[r'(?:Which company|Who) does (.+) work for\?',r'Where does (.+) work\?'],
 'P136':[r'What (?:(?:instrument|role) )?does (.+) play\?',r'What (?:kind|type) of music does (.+) play\?',r'What genre is (.+) associated with\?'],
 'P1376':[r'(?:What (?:country|state)|Which country) is (.+) the capital of\?',r'What is the capital of (.+)\?',r'What country or region has (.+) as its capital\?'],
 'P138':[r'(?:What|Who) is (.+) named after\?',r'After whom is (.+) named\?'],
 'P140':[r'(?:What|Which) (?:religion|faith|ideology) (?:does|did) (.+) follow\?',
         r'(?:What|Which)(?: (?:religion|faith))? (?:is|was) (.+) a follower of\?',
         r'Which religion was followed by (.+)\?',r'What does (.+) follow\?',r'Which religion follows (.+)\?'],
 'P159':[r'Where is (.+) headquartered\?',r'Where is (.+) headquarters located\?'],
 'P176':[r'Who produce[sd] (.+)\?',r'Which company produces (.+)\?'],
 'P178':[r'Who (?:developed|develops) (.+)\?'],
 'P19':[r'Where was (.+) born\?',r'What city was (.+) born in\?'],
 'P190':[r'(?:What|Which) city is (.+) a twin city of\?',r'What is (.+) a twin city of\?',
         r"What is (.+)'s twin city\?",r'(?:What is|Which city is) the twin city of (.+)\?'],
 'P20':[r'(?:When|In what year|Where) did (.+) die\?'],
 'P264':[r'What does (.+) represent\?',r'What is represented by (.+)\?',r'Who is (.+) represented by\?',
         r'Who is represented by (.+)\?',r'Who is representing (.+)\?',r'Who represents (.+)\?',r'Which record label is (.+) associated with\?']}


def subject_for(item):
    relation=item['relation'].split('(')[0]
    question=item['messages'][-1]['content'].strip()
    found=[match[1] for pattern in SUBJECT_PATTERNS.get(relation,[])
           if (match:=re.fullmatch(pattern,question))]
    unique=set(found)
    return next(iter(unique)) if len(unique)==1 else None


def group_for(item):
    subject = subject_for(item)
    if subject is None:
        raise ValueError(f"Unrecognized subject template: {item['item_id']}")
    key = item['relation'].split('(')[0] + ':' + normalized_label(subject)
    return sha256(key.encode()).hexdigest()


class UpstreamTuples:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        self.rows={}
        self.patterns={}
        for relation,entry in self.manifest['files'].items():
            if 'sha256' not in entry: continue
            path=self.directory/f'{relation}.jsonl'
            if sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Upstream source hash mismatch')
            by_subject=defaultdict(list)
            for row in map(json.loads,path.read_text(encoding='utf-8').splitlines()):
                by_subject[row['sub_label']].append(row)
            self.rows[relation]=by_subject
            self.patterns[relation]=defaultdict(list)
            for subject in by_subject:
                self.patterns[relation][normalized_label(subject)].append(subject)

    def align(self,item):
        relation=item['relation'].split('(')[0]
        extracted=subject_for(item)
        longest=self.patterns.get(relation,{}).get(normalized_label(extracted),[]) if extracted else []
        if len(longest)!=1:
            return {'status':'subject_unresolved','subject_candidates':longest,'independent_fact_verification':False}
        subject=longest[0]
        rows=self.rows[relation][subject]
        labels={normalized_label(row['obj_label']) for row in rows}
        matches=all(normalized_label(ref) in labels for ref in item['references'])
        return {'status':'historical_tuple_aligned' if matches else 'reference_not_in_filtered_upstream',
                'subject':subject,'matching_upstream_uuids':[row['uuid'] for row in rows
                    if normalized_label(row['obj_label']) in {normalized_label(ref) for ref in item['references']}],
                'available_upstream_objects':sorted({row['obj_label'] for row in rows}),
                'independent_fact_verification':False,
                'limitation':'Filtered historical tuples use string labels, not resolved entity IDs; an absent tuple is not evidence of a false reference.'}


def repair_question(item):
    """Only recognized relation/template pairs are rewritten; references never enter the prompt."""
    question=item['messages'][-1]['content']
    q=question.strip()
    relation=item['relation'].split('(')[0]
    rules=[]
    if relation=='P20':
        if re.fullmatch(r'Where did .+ die\?',q): return question,None,None
        rules=[(r'(?:When|In what year) did (.+) die\?',lambda s:f'Where did {s} die?','time_to_place_of_death')]
    elif relation=='P136':
        if re.fullmatch(r'What genre is .+ associated with\?',q): return question,None,None
        if re.fullmatch(r'What (?:kind|type) of music does .+ play\?',q):
            return question,None,None
        rules=[(r'What (?:(?:instrument|role) )?does (.+) play\?',lambda s:f'What genre is {s} associated with?','explicit_genre_relation')]
    elif relation=='P1376':
        if not q.startswith('What is the capital of '): return question,None,None
        rules=[(r'What is the capital of (.+)\?',lambda s:f'What country or region has {s} as its capital?','restore_capital_of_direction')]
    elif relation=='P264':
        if re.fullmatch(r'Which record label is .+ associated with\?',q): return question,None,None
        build=lambda s:f'Which record label is {s} associated with?'
        rules=[(r'What does (.+) represent\?',build,'explicit_record_label_relation'),
               (r'What is represented by (.+)\?',build,'explicit_record_label_relation'),
               (r'Who is (.+) represented by\?',build,'explicit_record_label_relation'),
               (r'Who is represented by (.+)\?',build,'explicit_record_label_relation'),
               (r'Who is representing (.+)\?',build,'explicit_record_label_relation'),
               (r'Who represents (.+)\?',build,'explicit_record_label_relation')]
    else:
        return question,None,None
    for pattern,build,name in rules:
        match=re.fullmatch(pattern,q)
        if match:
            subject=match[1]
            return build(subject),name,subject
    return question,'unrecognized_affected_template',None


def parse_choice_only(text,options):
    answer=text.strip()
    if answer not in options:
        return {'valid':False,'choice':None,'confidence':None,'reason':None,'error':'not_one_bare_legal_option'}
    return {'valid':True,'choice':options[answer],'confidence':None,'reason':None,'error':None}


def parse_structured(text,options):
    match=re.fullmatch(r'\s*Choice:\s*([^;\s]+);\s*Confidence:\s*(0(?:\.\d+)?|1(?:\.0+)?);\s*Reason:\s*([^\r\n]+?)\s*',text)
    if not match or match[1] not in options:
        return {'valid':False,'choice':None,'confidence':None,'reason':None,'error':'invalid_structured_contract'}
    reason=match[3].strip()
    if '<' in reason or '>' in reason or not re.search(r'[A-Za-z]',reason):
        return {'valid':False,'choice':None,'confidence':None,'reason':reason,'error':'placeholder_or_empty_reason'}
    if re.search(r'\b(?:Choice|Confidence|Reason)\s*:',reason):
        return {'valid':False,'choice':None,'confidence':None,'reason':reason,'error':'repeated_structured_fields'}
    return {'valid':True,'choice':options[match[1]],'confidence':float(match[2]),'reason':reason,'error':None}
