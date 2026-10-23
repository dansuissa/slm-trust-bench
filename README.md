# Small Language Model Trustworthiness

Code and development results for a study of factual answering, abstention, and output reliability in language models with fewer than 10 billion parameters.

The current release contains **26 completed CPU conditions and 1,216 generation calls** on SmolLM2-360M-Instruct and Qwen2.5-0.5B-Instruct. The experiments identified question/reference mismatches, questionnaire example copying, and substantial sensitivity to response format. They are development studies, not completed reproductions of the selected papers or a final model ranking. Semantic correctness and abstention judgments remain unannotated.

## Start here

| Document | Purpose |
| --- | --- |
| [Results](docs/results.md) | Completed measurements, corrections, and limits on interpretation |
| [Methods](docs/methods.md) | Generation, calibration, grouping, and cost definitions |
| [Paper selection](docs/paper_selection.md) | Review of the six starting papers and the three selected implementations |
| [Reproduction guide](docs/reproduction.md) | Installation, data preparation, inference, and analysis commands |
| [Annotation protocol](docs/annotation.md) | Grading rules and commands for context-answerability review |
| [Study design](docs/study_design.md) | Research questions and criteria for the main study |
| [External sources](THIRD_PARTY.md) | Attribution and redistribution boundaries |

## Quick start

Use Python 3.12. Run these commands from the repository root. The accounting check requires no model weights or external data.

```console
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in PowerShell. Then:

```console
python -m pip install -e .
trustbench smoke --out tmp/accounting
```

To prepare the inputs without running a model:

```console
python -m pip install -e '.[data]'
python -m trustbench.fetch
trustbench prepare
trustbench validate
python -m trustbench.protocols
```

The fetcher checks pinned SHA-256 hashes before saving. Data preparation reads prompt literals from upstream source files without executing those files. Downloaded material and local annotations are ignored by Git. Review the [source terms](THIRD_PARTY.md) before use.

For CPU inference, install the [documented runtime](docs/reproduction.md#cpu-runtime), then run one configuration per process:

```console
python -m trustbench.cpu --config configs/repair_factual_qwen05_repaired.json --out runs/factual_qwen_repaired
```

This command runs a six-question diagnostic with 12 generation calls. It does not produce adjudicated factual accuracy. Changed prompts or decoding settings require a new run directory. Resume is explicit and rejects changes to data, configuration, source code, or the recorded inference environment.

## What is included

- Independent data adapters, serial inference, per-draw recovery, calibration, and evaluation code in `src/trustbench/`.
- The 26 completed CPU configurations and exact input reconstruction recipes.
- Aggregate results with data, code, model, and output hashes in `results/`.
- Tests for denominator accounting, calibration boundaries, data leakage, recovery, likelihoods, downloads, and annotation validation.

Raw questionnaires, passages, upstream code, model weights, raw generations, and unfinished annotation packets are not distributed. Historical runs used the source hashes recorded in `results/completed_runs.json`. The organized package has additional modules and different cache paths; its source fingerprint is intentionally different. It cannot resume a historical run under a new fingerprint.

The core study will separate answer quality from the decision to answer. A model that always refuses, always repeats one choice, or only follows a constrained format must not appear trustworthy through a favorable denominator. See the [study design](docs/study_design.md) for the planned comparisons and the evidence required before making broader claims.

## License

Original code and documentation are MIT licensed. External inputs retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
