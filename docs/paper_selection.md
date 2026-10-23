# Paper selection and reproduction scope

The selection favors a clear measurement target, substantive public implementation, a tractable initial experiment, and a useful failure mode to investigate. Public code availability and an unrestricted redistribution license are separate questions. Repository revisions are pinned in `manifest.json`; downloaded source code is not redistributed.

## Selected papers

### FactTest

[Paper, ICML 2025](https://proceedings.mlr.press/v267/nie25a.html) | [Implementation](https://github.com/fannie1208/FactTest/tree/2f7bb87530d8321423304330964b70e50b621307)

FactTest frames factuality screening as a hypothesis test, calibrating a confidence-score threshold to control false acceptance conditional on an incorrect candidate. Its finite-sample treatment provides a precise target and makes calibration size, ties, and distribution assumptions testable. It is the strongest anchor for the factuality component.

The first original-model reproduction should target one ParaRel vanilla-entropy experiment with OpenLLaMA-3B. Preserve the released data and answer rule in a comparison branch, and separately evaluate corrected questions and disjoint groups. Document paper/release differences in sampling defaults, candidate versus scoring prompt spacing, and period-based answer extraction. Establish a table-level match before expanding to other scores or datasets. The present sub-billion CPU pilots change model and decoding protocol and do not reproduce that result.

### Investigating Value-Reasoning Reliability in Small Large Language Models

[Paper, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.395/) | [Implementation](https://github.com/Giovanna-SH/Value_Reasoning_Reliability/tree/0bbf0c5db2b7c8669a8a9d43cfd739a75d17bd84)

This paper distinguishes repeated-prompt consistency, robustness to equivalent formulations, interaction stability, and open-ended consistency. The released PVQ questionnaire is a small starting point for examining whether parsed stability reflects model behavior or the elicitation format.

Begin an original-model repetition comparison with Qwen2.5-7B-Instruct or Mistral-7B-Instruct-v0.3, using the literal released prompt and recorded checkpoint defaults. Keep decoding, template, special-token, and generation-prefix differences explicit. Do not equate an explicitly sampled CPU extension with the released-default protocol. The current diagnostics found copied example wording and parser failures; larger repetition runs are held. PVQ preferences have no objective factual or moral correctness label. INVP additionally contains missing definition mappings in the audited adapter and is excluded from the initial model study.

### AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions

[Paper](https://arxiv.org/abs/2506.09038) | [Implementation](https://github.com/facebookresearch/AbstentionBench/tree/e29184174c69bc95139b8c33dcca09aacab9442a)

Added beyond the six starting papers, AbstentionBench provides substantive evaluation code and several distinct reasons for abstaining. A bounded SQuAD 2.0 comparison gives a tractable context-answerability task alongside closed-book factuality. It is the practical replacement for a starting paper without a runnable public release.

An original-model comparison must preserve the chosen paper checkpoint, sampling, token budget, dataset subset, and judge protocol. The published judge prompt includes reference information and the answerability label; assess agreement against a separate human-reviewed sample before using it in the main benchmark. The current greedy, 64-token, sub-billion development slice changes several of those settings. It covers context answerability, not all AbstentionBench scenarios or general safety.

## Other starting papers

| Paper | Scientific use | Code assessment and decision |
| --- | --- | --- |
| [SLM-Bench: A Comprehensive Benchmark of Small Language Models on Environmental Impacts](https://aclanthology.org/2025.findings-emnlp.1165/) | Useful precedent for reporting accuracy, computation, and consumption together. Shared hardware and measurement definitions matter. | [Public repository](https://github.com/HiveIntel/SLM-Bench) contains implementation files. The audited release does not provide a clean, directly reusable reproduction path for this trustworthiness study. Use its measurement lessons; do not execute its training script as the benchmark harness. Latency/RSS here do not reproduce its energy or carbon results. |
| [Towards Trustworthy Lexical Simplification: Exploring Safety and Efficiency with Small LLMs](https://aclanthology.org/2025.inlg-main.15/) | Valuable example of automatic task metrics missing harmful simplifications; supports manual error analysis and probability-based filtering as related work. | The linked [repository](https://github.com/ahaya3776/safe-efficient-ls) contains only a README promising a later update in the inspected release; the public page was rechecked on 23 October 2026. It does not satisfy the runnable-code requirement. The paper's training/distillation and multilingual scope would also enlarge this project's workload. |
| [Smaller Large Language Models Can Do Moral Self-Correction](https://aclanthology.org/2025.trustnlp-main.5/) | Motivates a later, carefully scoped training-free intervention and highlights alignment/instruction effects. | A corresponding runnable author release was not verified in the initial audit. Treat it as related work, not an implementation-backed reproduction selection. Its social-stereotype endpoints also differ from factual correctness. |
| [Can LLMs Refuse Questions They Do Not Know?](https://arxiv.org/abs/2510.01782) | Separates knowledge-aware refusal from raw refusal frequency; useful when defining over-refusal and knowledge estimation. | The linked [Refusal-Index repository](https://github.com/WENBOPAN/Refusal-Index) returned 404 when rechecked on 23 October 2026. Do not advertise a successful code-based reproduction. A two-pass refusal study could be reconsidered if an executable release becomes available. |

The three selections establish complementary measurement methods. Reproducing their entire experimental programs is outside the initial scope. A defensible first milestone is one documented original-model result from each feasible selected protocol, followed by a narrower shared factuality/answerability study.
