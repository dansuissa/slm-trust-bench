# Development results

Completed on 23 October 2026. These measurements describe the development protocols. They do not establish semantic accuracy, general safety, or reproduction of an original paper result.

The full journal audit verified **26 conditions and 1,216 generation calls**. Calls include repeated questions and auxiliary samples; they are not independent examples. All completed conditions are listed with configuration, data, source, and output hashes in [completed_runs.json](../results/completed_runs.json). Historical integration checks on a different tiny model are excluded.

## Context answerability

Each model answered the same 400 development questions: 200 answerable and 200 unanswerable. No human correctness or abstention labels have been completed.

| Model | Responses | Empty | Length-capped | Observed peak RSS (GiB) | Sum of call timers (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 400 | 0 | 64 | 2.92 | 28.26 |
| SmolLM2-360M-Instruct | 400 | 0 | 18 | 2.25 | 18.66 |

Call timers exclude loading, seed setup, likelihood aggregation, and journal I/O. RSS is sampled process memory. These are shared-host CPU measurements, not complete service latency or energy estimates. The differing length-cap rates warrant a preregistered budget comparison before a final model ranking.

## Factual question audit

Across 600 development/calibration candidates, 58 questions required a type or relation-direction correction, and 74 received a genre or record-label clarification. References were preserved. These 132 interventions do not mean that 132 reference facts were false.

The changes include asking where a person died when the reference is a place, restoring the direction of a capital-of relation, and specifying genre or record label instead of an ambiguous wording. Regrouping by relation and subject removed one development/calibration overlap and quarantined two calibration records with different references, leaving 497 calibration candidates.

A paired diagnostic reran six selected questions on each model before and after wording changes. All 24 greedy candidates were nonempty; each six-question condition had zero normalized exact and zero released substring reference matches. This targeted sample is too small and selective to estimate factual accuracy or the effect of repair. It contains potential reference-coverage and specificity issues as well as model errors. Independent adjudication remains necessary.

Historical ParaRel tuple alignment found 425/600 reference sets aligned by subject label, 168 unresolved subjects, and seven reference sets absent from the filtered upstream tuples. Absence is not evidence that a reference is false. This alignment is not independent factual verification.

## Questionnaire format validation

Eight development questions per model were sampled twice in each condition. Four questions had already been inspected; four were fresh development questions. All conditions were exploratory. The original/released, structured, and number-only contracts require different content, so their parse counts are not directly comparable measures of reasoning quality.

| Model | Contract | Successful parses | Fresh-question parses | Released example phrase present |
| --- | --- | ---: | ---: | ---: |
| SmolLM2-360M | released | 11/16 | 7/8 | 13/16 |
| Qwen2.5-0.5B | released | 13/16 | 5/8 | 0/16 |
| SmolLM2-360M | neutral_structured | 2/16 | 0/8 | 0/16 |
| Qwen2.5-0.5B | neutral_structured | 9/16 | 5/8 | 0/16 |
| SmolLM2-360M | choice_only | 8/16 | 4/8 | 0/16 |
| Qwen2.5-0.5B | choice_only | 16/16 | 8/8 | 0/16 |

Removing the filled example eliminated the measured copied phrase in these validation outputs. It did not resolve structured-format compliance, especially for SmolLM2. The number-only condition improved parse validity for Qwen in this small sample, but omits both reasoning and confidence. No semantic quality or preference-validity improvement has been established. The initial four-question factorial controls remain in the run index; a larger repetition study was not launched.

## Calibration implementation check

The synthetic diagnostic completed 180,000 replications across 18 cells: six calibration sizes and three known score distributions. It checks strict-threshold behavior, insufficient calibration data, ties, and an explicit distribution-shift counterexample. It uses no model responses. Full results are in [calibration_synthetic.json](../results/calibration_synthetic.json).

## What remains unresolved

- Human grading of context responses and factual references/answers.
- A fixed comparison of output budgets, with semantic labels and matched questions.
- Original-paper model/protocol reproductions, including larger small models.
- Confirmatory comparisons on held-out data, with group-level uncertainty and declared primary endpoints.

Original results remain valid records of their stated protocols. Changed question wording or response contracts need new conditions; they do not require rerunning unaffected context outputs. A higher token budget is also a new protocol and should be reported alongside the original budget.
