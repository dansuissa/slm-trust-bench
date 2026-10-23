"""Render the descriptive results table from verified aggregate artifacts."""
import argparse
import json
from pathlib import Path


def render(directory):
    directory = Path(directory)
    audit = json.loads((directory / 'completed_runs.json').read_text(encoding='utf-8'))
    validation = json.loads((directory / 'format_validation.json').read_text(encoding='utf-8'))
    repairs = json.loads((directory / 'data_audit.json').read_text(encoding='utf-8'))
    synthetic = json.loads((directory / 'calibration_synthetic.json').read_text(encoding='utf-8'))
    lines = [
        '# Development results', '',
        'Completed on 23 October 2026. These measurements describe the development protocols. They do not establish semantic accuracy, general safety, or reproduction of an original paper result.', '',
        f"The full journal audit verified **{audit['completed_conditions']} conditions and {audit['generation_calls']:,} generation calls**. Calls include repeated questions and auxiliary samples; they are not independent examples. All completed conditions are listed with configuration, data, source, and output hashes in [completed_runs.json](../results/completed_runs.json). Historical integration checks on a different tiny model are excluded.", '',
        '## Context answerability', '',
        'Each model answered the same 400 development questions: 200 answerable and 200 unanswerable. No human correctness or abstention labels have been completed.', '',
        '| Model | Responses | Empty | Length-capped | Observed peak RSS (GiB) | Sum of call timers (min) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in audit['runs']:
        if row['run'].startswith('c6_') and row['run'].endswith('_development'):
            lines.append(f"| {row['model']['id'].split('/')[-1]} | {row['responses']} | {row['empty_candidates']} | {row['candidate_quality'].get('length_capped', 0)} | {row['peak_process_rss_bytes'] / 2**30:.2f} | {row['generation_timer_seconds'] / 60:.2f} |")
    lines += [
        '', 'Call timers exclude loading, seed setup, likelihood aggregation, and journal I/O. RSS is sampled process memory. These are shared-host CPU measurements, not complete service latency or energy estimates. The differing length-cap rates warrant a preregistered budget comparison before a final model ranking.', '',
        '## Factual question audit', '',
        f"Across 600 development/calibration candidates, {repairs['question_intervention_types']['semantic_type_or_direction_repair']} questions required a type or relation-direction correction, and {repairs['question_intervention_types']['relation_disambiguation']} received a genre or record-label clarification. References were preserved. These {repairs['question_repairs_total']} interventions do not mean that 132 reference facts were false.", '',
        'The changes include asking where a person died when the reference is a place, restoring the direction of a capital-of relation, and specifying genre or record label instead of an ambiguous wording. Regrouping by relation and subject removed one development/calibration overlap and quarantined two calibration records with different references, leaving 497 calibration candidates.', '',
        'A paired diagnostic reran six selected questions on each model before and after wording changes. All 24 greedy candidates were nonempty; each six-question condition had zero normalized exact and zero released substring reference matches. This targeted sample is too small and selective to estimate factual accuracy or the effect of repair. It contains potential reference-coverage and specificity issues as well as model errors. Independent adjudication remains necessary.', '',
        'Historical ParaRel tuple alignment found 425/600 reference sets aligned by subject label, 168 unresolved subjects, and seven reference sets absent from the filtered upstream tuples. Absence is not evidence that a reference is false. This alignment is not independent factual verification.', '',
        '## Questionnaire format validation', '',
        'Eight development questions per model were sampled twice in each condition. Four questions had already been inspected; four were fresh development questions. All conditions were exploratory. The original/released, structured, and number-only contracts require different content, so their parse counts are not directly comparable measures of reasoning quality.', '',
        '| Model | Contract | Successful parses | Fresh-question parses | Released example phrase present |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for name, row in validation['value'].items():
        _, _, model, condition = name.split('_', 3)
        label = {'smol360':'SmolLM2-360M', 'qwen05':'Qwen2.5-0.5B'}[model]
        lines.append(f"| {label} | {condition} | {row['contract_parse_successes']}/{row['draws']} | {row['fresh_development_parse_successes']}/{row['fresh_development_draws']} | {row['verbatim_example_reason_phrase']}/{row['draws']} |")
    lines += [
        '', 'Removing the filled example eliminated the measured copied phrase in these validation outputs. It did not resolve structured-format compliance, especially for SmolLM2. The number-only condition improved parse validity for Qwen in this small sample, but omits both reasoning and confidence. No semantic quality or preference-validity improvement has been established. The initial four-question factorial controls remain in the run index; a larger repetition study was not launched.', '',
        '## Calibration implementation check', '',
        f"The synthetic diagnostic completed {sum(cell['trials'] for cell in synthetic['cells']):,} replications across {len(synthetic['cells'])} cells: six calibration sizes and three known score distributions. It checks strict-threshold behavior, insufficient calibration data, ties, and an explicit distribution-shift counterexample. It uses no model responses. Full results are in [calibration_synthetic.json](../results/calibration_synthetic.json).", '',
        '## What remains unresolved', '',
        '- Human grading of context responses and factual references/answers.',
        '- A fixed comparison of output budgets, with semantic labels and matched questions.',
        '- Original-paper model/protocol reproductions, including larger small models.',
        '- Confirmatory comparisons on held-out data, with group-level uncertainty and declared primary endpoints.', '',
        'Original results remain valid records of their stated protocols. Changed question wording or response contracts need new conditions; they do not require rerunning unaffected context outputs. A higher token budget is also a new protocol and should be reported alongside the original budget.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--out', type=Path, default=Path('docs/results.md'))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.results), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
