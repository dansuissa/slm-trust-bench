import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from trustbench.repairs import repair_question, subject_for, parse_choice_only, parse_structured, UpstreamTuples, group_for




def item(question,relation='P20(place of death)',reference='SOURCE_REFERENCE'):
    return {'messages':[{'role':'user','content':question}],'relation':relation,'references':[reference],'item_id':'toy:one'}


class QuestionRepairTests(unittest.TestCase):
    def test_time_and_year_become_place_without_inserting_gold(self):
        for q in ['When did Person X die?','In what year did Person X die?']:
            source=item(q,reference='DO_NOT_LEAK_THIS_REFERENCE')
            self.assertEqual(repair_question(source)[0],'Where did Person X die?')
            self.assertEqual(source['messages'][0]['content'],q)
            self.assertNotIn(source['references'][0],repair_question(source)[0])

    def test_genre_does_not_imply_music_for_books_or_other_works(self):
        source=item('What role does Animal Farm play?','P136(genre)','satire')
        self.assertEqual(repair_question(source)[0],'What genre is Animal Farm associated with?')
        typed=item('What type of music does Band X play?','P136(genre)')
        self.assertIsNone(repair_question(typed)[1])

    def test_capital_direction_and_correct_direction_are_distinguished(self):
        q=item('What is the capital of London?','P1376(capital of)','England')
        self.assertEqual(repair_question(q)[0],'What country or region has London as its capital?')
        good=item('What country is London the capital of?','P1376(capital of)')
        self.assertEqual(repair_question(good)[0],good['messages'][0]['content'])

    def test_inverted_record_label_wording_recovers_subject(self):
        source=item('Who is represented by Jimmy & Johnny?','P264(record label)','Label X')
        self.assertEqual(subject_for(source),'Jimmy & Johnny')
        self.assertEqual(repair_question(source)[0],'Which record label is Jimmy & Johnny associated with?')

    def test_corrected_templates_are_idempotent(self):
        for source in [item('When did Person X die?'),item('What instrument does Artist X play?','P136(genre)'),
                       item('Who represents Singer X?','P264(record label)'),item('What is the capital of London?','P1376(capital of)')]:
            q,_,_=repair_question(source)
            corrected=copy.deepcopy(source);corrected['messages'][0]['content']=q
            self.assertEqual(repair_question(corrected)[0],q)
            self.assertIsNone(repair_question(corrected)[1])
            self.assertEqual(subject_for(corrected),subject_for(source))

    def test_grouping_catches_paraphrases_and_does_not_use_reference(self):
        a=item('Which city is Budapest a twin city of?','P190(twinned administrative body)','Lviv')
        b=item("What is Budapest's twin city?",'P190(twinned administrative body)','Frankfurt')
        self.assertEqual(group_for(a),group_for(b))
        b['relation']='P1376(capital of)';b['messages'][0]['content']='What country is Budapest the capital of?'
        self.assertNotEqual(group_for(a),group_for(b))

    def test_unrecognized_affected_template_is_explicit(self):
        self.assertEqual(repair_question(item('Tell me where X died.'))[1],'unrecognized_affected_template')

    def test_upstream_absence_is_not_classified_as_a_false_reference(self):
        scratch=Path(__file__).resolve().parents[2]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            p=Path(folder)
            data=json.dumps({'sub_label':'Person X','obj_label':'City A','uuid':'u'})+'\n'
            (p/'P20.jsonl').write_text(data,encoding='utf-8',newline='\n')
            (p/'manifest.json').write_text(json.dumps({'files':{'P20':{'sha256':sha256(data.encode()).hexdigest()}}}))
            index=UpstreamTuples(p)
            result=index.align(item('When did Person X die?',reference='Another possibly valid place'))
            self.assertEqual(result['status'],'reference_not_in_filtered_upstream')
            self.assertFalse(result['independent_fact_verification'])
            (p/'P20.jsonl').write_text('corrupted')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): UpstreamTuples(p)


class NeutralFormatTests(unittest.TestCase):
    options={str(i):str(i) for i in range(1,7)}

    def test_choice_only_accepts_only_one_legal_token_string(self):
        self.assertTrue(parse_choice_only(' 4\n',self.options)['valid'])
        for text in ['4 or 5','4.5','I choose Option 4','7','1\n2','']:
            self.assertFalse(parse_choice_only(text,self.options)['valid'],text)

    def test_structured_fields_have_separate_parsed_confidence(self):
        result=parse_structured('Choice: 4; Confidence: 0.75; Reason: This description fits my answer.',self.options)
        self.assertEqual(result['choice'],'4');self.assertEqual(result['confidence'],.75)
        self.assertTrue(result['valid'])

    def test_invalid_confidence_placeholders_extra_lines_and_duplicate_fields_rejected(self):
        for text in ['Choice: 4; Confidence: 1.1; Reason: A reason.',
                     'Choice: 7; Confidence: 0.5; Reason: A reason.',
                     'Choice: 4; Confidence: nan; Reason: A reason.',
                     'Choice: 4; Confidence: 0.5; Reason: <one sentence>',
                     'Choice: 4; Confidence: 0.5; Reason: A reason.\nChoice: 2',
                     'Choice: 4; Confidence: 0.5; Reason: A reason; Choice: 2']:
            self.assertFalse(parse_structured(text,self.options)['valid'],text)
