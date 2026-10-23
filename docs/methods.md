# Methods

## Experimental unit and provenance

Each configuration pins a model commit, data file, decoding parameters, prompt rendering, seed, context limit, and repetition count. Runs record the environment, loaded parameter count, resolved generation defaults, source hashes, and rendered-prompt hashes. The CPU runner writes and synchronizes a journal after every generation call. Only a trailing partial record can be recovered; the original bytes are preserved. Interior corruption and duplicate draw identities are rejected.

Repeated generations of one question are dependent measurements. A passage may also contain several questions. Questions are the sampling unit for factual QA; shared passages are the grouping unit for SQuAD. Bootstrap intervals resample groups, not individual draws. The current aggregates are descriptive, without model comparison significance claims.

The two measured checkpoints have 361,821,120 and 494,032,768 loaded parameters respectively. The broader inclusion criterion is fewer than 10 billion total parameters, including all experts for a mixture-of-experts model. Nominal model names do not replace parameter counts.

## Data partitions

The pinned FactTest release contains 5,575 calibration, 5,584 in-distribution test, and 11,180 shifted-test records. There are 175 question groups shared between calibration and in-distribution test. The independent adapter merges repeated identical question/reference records, quarantines groups with different reference sets, and excludes calibration-shared test groups. These operations yield 5,294 calibration and 5,185 in-distribution test questions. Different reference sets are not assumed to be false labels.

The CPU work used 100 development and 500 calibration candidates drawn deterministically from the cleaned calibration source. A separate exploratory test allocation has 300 questions; the remaining 4,885 cleaned test questions are reserved from exploratory tuning. The repair builder inspected development and calibration views only. A test set being reserved from this project's tuning does not establish that a model never encountered it during training.

Question-template corrections use the relation and extracted subject, without inserting reference answers. The revised grouping key is relation plus case-folded, whitespace-normalized subject label. This removes one calibration overlap with development and quarantines two calibration records sharing a subject/relation but different references, leaving 497 calibration candidates. String labels are not entity identifiers; unresolved ambiguity still requires review.

SQuAD 2.0 validation contains 11,873 questions grouped into 1,204 passages. A deterministic hash of each passage assigns approximately 20% of passage groups to development, 30% to calibration, and 50% to test. Counts are 2,239, 3,762, and 5,872 questions. The measured development slice has 200 answerable and 200 unanswerable questions. Its precision and overall accuracy depend on that imposed prevalence.

## Prompts and generation

The CPU experiments use FP32, batch size one, one compute thread, and one resident model. Native chat rendering uses the pinned tokenizer's generation prompt and disables additional special-token insertion. The raw rendering ablation receives the same message content. Tokenizer templates may add default system text, so the comparison does not isolate special tokens alone.

Factual candidates are greedy. The rendering pilot uses three stochastic auxiliary answers at temperature 0.7, top-p 1, top-k 0, with a 24-token cap. Its score is the negative empirical entropy of distinct stripped answer strings. Empty candidates or samples are retained and assigned a score below the valid score range. Three samples are an engineering diagnostic, not a precise uncertainty estimate. The repair pilot retains one auxiliary answer for runner compatibility and makes no entropy/calibration claim.

PVQ conditions sample at temperature 0.7, top-p 1, top-k 0, with two repetitions. The original and structured conditions have a 96-token cap; the number-only condition has an 8-token cap. Context-answerability candidates are greedy with a 64-token cap. A length cap is reported even when the visible text appears sufficient to answer. Empty outputs are separate from abstentions.

The released questionnaire, neutral structured answer, and number-only answer impose different tasks. Their parsers respectively require a single released choice phrase, one legal choice with confidence in [0,1] and a non-placeholder reason, and a single bare legal number. Parse success across these contracts is not a common measure of reasoning quality. Repetition statistics report the invalid-draw denominator and the subset with all required valid repetitions. A copied example cannot establish stable preferences.

## Correctness and abstention

The runner's factual `correct` field is an automatic normalized reference-string match. The historical substring field implements a separate released string rule. Neither is independent semantic adjudication. References may omit aliases, other valid answers, time qualifiers, or a different valid level of specificity. Calibration for the main study must use reviewed correctness labels under a declared rubric.

Context abstention is judged against information in the supplied passage. A concrete hedged answer is still an answer. An explicit statement that the passage does not support an answer is abstention. Contradictory mixed answers, empty responses, and unclear responses remain distinct. This task does not establish harmful-request refusal or knowledge-aware refusal over pretraining knowledge.

## Calibration

The FactTest target implemented here is **P(accept | incorrect)**. Selective error is **P(incorrect | accepted)** and is reported separately. These conditional probabilities have different denominators.

For incorrect calibration scores sorted increasingly, the procedure selects the smallest order statistic k satisfying P(Binomial(n, 1-alpha) >= k) <= delta. Acceptance requires a score strictly greater than the selected threshold. If no finite order statistic qualifies, the policy rejects all candidates. At alpha = 0.05 and delta = 0.01, at least 90 incorrect calibration observations are required for a finite threshold. Tied scores can still produce zero coverage.

The finite-sample guarantee requires appropriate independent sampling and an unchanged applicable score/label distribution. Software checks cannot establish these assumptions. The explicit published-reproduction mode preserves original duplication for comparison but marks the independence assumption as unverified. It does not restore a guarantee. Synthetic diagnostics check implementation behavior under known distributions; they are neither a new theorem nor evidence of a guarantee under arbitrary distribution shift.

## Cost measurements

Per-call generation timers include prompt tokenization, generation, and decoding. They exclude model loading, seed setup, likelihood aggregation, and journal I/O. Run timestamps describe a wider elapsed interval and can include interruption time. Report these quantities separately. They are not full service latency, energy consumption, or carbon emissions.

Process resident memory is polled at 50 ms intervals and can miss brief peaks. The 3.5 GiB process budget and 0.65 GiB host reserve are checks before and after generation, not operating-system memory limits. Generation can temporarily exceed a budget before a check. Shared-host contention, CPU model, thread count, prompt length, output length, and loading policy must accompany cost comparisons.

Candidate log likelihoods use raw, unprocessed next-token logits, not sampling-transformed scores. EOS is excluded from answer-only sums and means. These are score baselines, not calibrated probabilities that the answer is correct.
