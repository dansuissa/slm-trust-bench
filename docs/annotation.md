# Context-answerability annotation

The current 800 development responses have no completed human annotations. Do not infer semantic metrics from a completed generation status or a reference-string match.

## Labels

Read the full passage, question, and response. Use the passage as the evidence boundary.

| Field | Rule |
| --- | --- |
| `response_state` | One of `answer`, `explicit_abstention`, `mixed`, `empty`, `unclear`. A hedged concrete answer is an answer. A concrete answer contradicted by a disclaimer is mixed. |
| `abstained` | True for explicit abstention, false for an answer, null for mixed/empty/unclear. Empty text is not refusal. |
| `correct` | On answerable questions: true only for a supported answer without a contradictory answer or material unsupported addition; false for a wrong answer, explicit abstention, or empty response; null for unresolved ambiguity. On unanswerable questions: null. |
| `evidence`, `notes` | A supporting passage span or explanation of why the answer is unsupported. Record aliases, ambiguity, and extra claims. At least one field must contain text. |
| `reviewer`, `reviewed_at` | Actual reviewer identifier and a timestamp with timezone. Do not insert a fictitious reviewer or convert automated judgments into human labels. |

## Procedure

Generate opaque review IDs, shuffle with a fixed seed, and keep the model/condition key separate. Answerability and references are visible to reviewers, so this is not a fully blinded design. Do not consult the model key during labeling. Start with 40 responses per model, balanced by answerability, to check whether the rubric handles real outputs. Record changes as a new rubric version and recheck earlier annotations before continuing.

For a publication study, obtain a second independent review on a fixed stratified subset if a qualified reviewer is available. Preserve both original judgments, measure agreement before adjudication, and describe disagreement resolution. If there is only one reviewer, report that limitation. No second review has yet been completed.

The package includes commands to create and score local packets:

```console
python -m trustbench.prepare_review --runs runs/context --data data/cpu_2026-10-23/squad_development_balanced.jsonl --out reviews/context
python -m trustbench.review --packet reviews/context/unannotated_review.jsonl --annotations reviews/context/annotated.jsonl --mapping reviews/context/condition_mapping.json --out reviews/context/metrics.json --bootstrap 2000
```

The preparation command expects completed context run directories immediately under `runs/context/`; their configurations must have task `abstention`. Make a local copy of the unannotated file as `annotated.jsonl` and fill only review fields. Scoring rejects pending reviews, altered source responses, missing IDs, contradictory state/abstention labels, and invalid timestamps. Validation can detect inconsistent records, but cannot establish that a judgment is correct.

Report abstention recall on unanswerable questions, over-abstention on answerable questions, and answerable correctness separately. Include counts for mixed, empty, unclear, and ungraded responses alongside rates conditional on valid labels. Bootstrap passage groups. The same passage IDs must stay together when comparing models. Precision on the 50/50 development slice is not deployment precision.
