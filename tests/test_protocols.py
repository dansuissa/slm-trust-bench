from trustbench.protocols import transform
from trustbench.repairs import group_for
import unittest


class ProtocolTests(unittest.TestCase):
    def test_repair_regroups_without_reference_leak_or_source_mutation(self):
        row = {'item_id':'toy', 'group_id':'old', 'relation':'P20(place of death)',
               'messages':[{'role':'user','content':'When did Person X die?'}], 'references':['SECRET_REFERENCE']}
        revised = transform(row, 'factual_repaired', {})
        self.assertEqual(revised['messages'][-1]['content'], 'Where did Person X die?')
        self.assertNotIn('SECRET_REFERENCE', revised['messages'][-1]['content'])
        self.assertEqual(revised['group_id'], group_for(row))
        self.assertEqual(row['messages'][-1]['content'], 'When did Person X die?')

    def test_neutral_contract_uses_source_cells_and_no_released_example(self):
        row = {'item_id':'PVQ:toy', 'source_row':2, 'messages':[{'role':'user','content':'An example response.'}]}
        result = transform(row, 'value_validation_choice_only', {'PVQ:toy':('A question?', '1. First 2. Second')})
        self.assertIn('A question?', result['messages'][-1]['content'])
        self.assertNotIn('An example response', result['messages'][-1]['content'])
        self.assertEqual(result['pilot_role'], 'previously_inspected')
