# Main study design

This is a proposed design informed by development diagnostics. It is not a preregistration, and no confirmatory test results have been collected under it.

## Research question

How do small language models trade correct answering against inappropriate answering and unnecessary abstention, once question quality, grading, prompt rendering, and inference cost are controlled?

The intended contribution is an auditable comparison of **useful correct answers at a stated screening error target and measured cost**, supported by an analysis of measurement failures. A larger leaderboard alone would not establish a new result. Prompt sensitivity, sampling consistency, likelihood scoring, and cost-aware selection already have substantial prior work; none is claimed as a new method here.

In particular, [Second Guess](https://arxiv.org/abs/2605.25394) already studies parameter-free abstention through answer stability in small models. [When Should a Language Model Trust Itself?](https://arxiv.org/abs/2605.02915) directly compares self-verification with mean and summed answer likelihoods. [LLM Abstention Can Be a Prompt Artifact](https://arxiv.org/abs/2507.16199v9) studies refusal induced by the answer format. A claim that stability, likelihood baselines, or prompt-induced refusal is itself novel would therefore be unsupported. The potential contribution here is testing how validated question/label instruments and explicit cost accounting change substantive small-model conclusions. Whether those changes are sufficiently distinct and important remains an empirical question.

Keep the core to closed-book factual QA and context answerability. Retain questionnaire consistency as a separate diagnostic study only if it helps explain an important measurement problem. Harmful-request safety, multilingual evaluation, domain adaptation, fine-tuning, and broad adversarial testing are outside this version's main claim.

## Questions and endpoints

| Question | Primary measurement | Supporting measurements |
| --- | --- | --- |
| Does a screening score preserve useful answers while controlling false acceptance among incorrect answers? | Correct accepted answers per original query at a declared alpha/delta policy | Coverage, selective error, false acceptance conditional on incorrectness, ungraded/failure counts, score cost |
| Can a model identify when the supplied evidence does not support an answer without refusing supported questions? | Unsupported-answer rate and over-abstention, reported separately | Answerable correctness, mixed/empty/unclear counts, answerability prevalence, passage-group intervals |
| Which findings depend on measurement choices? | Paired changes on the same base questions after a predeclared intervention | Reference adjudication, output truncation, parser validity, prompt rendering, grading disagreement |

Select at most two confirmatory contrasts after pilot annotation and before opening held-out outputs. For example, compare answer mean-log-likelihood with a fixed-size sampling score under one frozen generator. Fix the output budget and sampling count rather than selecting the best setting on test results. Avoid a large model-by-dataset-by-prompt-by-budget factorial design.

## Data and grading decisions

1. Independently review the 600 factual development/calibration candidates. Resolve relation direction, answer type, entity ambiguity, aliases, multiple valid answers, and time qualifiers. Preserve original labels and provenance alongside the adjudicated view. A historical tuple match is supporting provenance, not independent truth.
2. Validate the rubric on 40 context responses per model, balanced by answerability, then complete the 800-response development review. Obtain a second review on a fixed subset when possible. Publish the actual review coverage and disagreements.
3. Freeze eligibility and exclusion rules before final evaluation. Keep all paraphrases, answer variants, and shared passages within one split. Revised labels or prompts create a new protocol version.
4. Use a competency pilot before scaling a rare-entity factual task to tiny models. If useful correct answering is near zero after adjudication, add a clearly defined, more suitable factual set or change the model panel. Do not optimize a confidence method around a degenerate accuracy floor.
5. Add one further answerability scenario, such as false premises or conflicting evidence, only if it addresses a specific remaining hypothesis. Do not pool distinct abstention mechanisms into one unexplained safety score.

## Models and baselines

The measured CPU models establish runtime feasibility and expose instrument problems. They are not a representative sample of all small LLMs. Original-paper reproductions should include the relevant 3B or 7B checkpoint before making reproduction claims.

For the main comparison, target a small panel of roughly four checkpoints spanning a within-family size comparison and at least one second family. Pin exact revisions after allocation and competency checks. Report the selection rationale, release/instruction-training differences, parameter counts, precision, and access terms. Model-family effects are not causal parameter-scaling effects.

Include always-answer and always-abstain baselines, candidate answer quality without screening, answer likelihood sum/mean, and one fixed-budget sampling-consistency score. A random score can check whether selection adds information. Self-reported confidence is optional and requires its own calibration evaluation. Forced-choice decoding is a distinct elicitation task; it must not remove abstention from a refusal evaluation.

## Statistical plan

Freeze the generator, prompt, tokenizer, extraction rule, score, threshold rule, and grading protocol together. Report the number of incorrect independent calibration candidates and all-reject fallback frequency. If a finite threshold is infeasible, enlarge calibration or change the target explicitly; do not relax it after seeing test performance.

Use paired comparisons on shared question/passage groups. Estimate uncertainty with a group bootstrap and report undefined ratios when a denominator is zero. Separate generation variation, calibration-set variation, and grader disagreement. For alpha = 0.05 and delta = 0.01, 90 incorrect calibration examples merely make a finite order statistic possible; this is not a sample-size justification for precise empirical error estimates.

Choose final sample size from pilot discordant-pair rates, group counts, the desired interval width, and the smallest meaningful effect. About 400 independent Bernoulli observations can still have a roughly five-percentage-point 95% margin near 50%; passage clustering can make it wider. Repeated draws do not substitute for additional independent questions. Declare primary contrasts and treat the rest as exploratory, with an explicit multiple-comparison policy where needed.

## Compute and schedule

The measured CPU host has four logical CPUs, approximately 5.5 GiB RAM, and no usable GPU in the inspected environment. One-thread serial runs fit the two measured sub-billion models. Remote CPU jobs avoid using the personal laptop for inference; the laptop does not need to remain awake once jobs run inside a persistent remote session. Long jobs still need to respect the host's course-use rules and account allocation.

A GPU with at least 24 GB VRAM is a practical request for one resident 3B or 7B model with bounded context and pilot batching. Weight-only FP16 storage is approximately 6 GB for 3B parameters and 14 GB for 7B, before cache, activations, runtime buffers, or a judge. Memory fit and throughput must be measured before promising runtime. Quantization is a separate experimental condition requiring fresh calibration.

| Period | Deliverable and decision |
| --- | --- |
| Late October to early November 2026 | Finish development annotation and reference adjudication; test a fixed output-budget comparison on a bounded paired subset. |
| November 2026 | Complete one tractable original-model/protocol reproduction per feasible selected paper; document any table discrepancy. Establish GPU throughput and memory. |
| December 2026 | Freeze the main hypotheses, model panel, grading, splits, baselines, and statistical plan after the competency pilots and closest-work review. |
| January to February 2027 | Run the held-out core study; complete the independent annotation audit and focused error analysis. |
| March 2027 | Run only ablations needed to explain the main findings; consider one training-free intervention if the core results are complete. |
| April 2027 | Write the paper, reproduce tables from frozen artifacts, review claims and data permissions, and archive the final release before the end-of-April deadline. |

Publication readiness requires more than a passing pipeline: a distinct contribution against the closest literature, trustworthy labels, appropriate controls, uncertainty estimates, independent review where feasible, and results that survive held-out evaluation. The present release provides the infrastructure and development evidence for those decisions.
