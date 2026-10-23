# Reproduction guide

## Installation

Use Python 3.12 and run from the repository root. Python 3.11 is allowed by package metadata but has not been used for the recorded model runs. Windows is supported for local checks; the measured inference environment is Linux. In PowerShell, use `.venv\Scripts\python.exe` if the environment is not activated.

```console
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1`.

```console
python -m pip install -e .
trustbench smoke --out tmp/accounting
```

The smoke command uses synthetic records. It is not a model result.

## Data

```console
python -m pip install -e '.[data]'
python -m trustbench.fetch
trustbench prepare
trustbench validate
python -m trustbench.protocols
python -m trustbench.question_audit --out data/question_audit
```

The first command fetches nine pinned external artifacts. SHA-256 mismatches fail before saving; changed existing artifacts are not overwritten. Preparation reconstructs published data and prompt literals locally, reports duplicate/overlap counts, and groups SQuAD by passage. Protocol preparation rebuilds the 13 exact input files used across the 26 completed CPU conditions and checks their frozen hashes, including historical line endings. The question audit separately reconstructs the 600 development/calibration corrections and review ledger; it never opens the reserved test views.

These steps do not grant reuse rights or establish reference truth. See [THIRD_PARTY.md](../THIRD_PARTY.md). Keep the generated `data/` directory local.

## CPU runtime

For Linux CPU-only PyTorch, install the wheel from its official index first:

```console
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[cpu,data]'
python -m unittest discover -s tests -v
```

The pinned historical inference stack uses Transformers 4.51.3, Accelerate 1.6.0, and FP32. Full per-run environment records are in `results/completed_runs.json`. The package pins the main runtime dependencies; it is not a lock for every transitive package on every operating system. Changes to the recorded environment invalidate resume. Runtime pinning preserves comparability and should be reassessed before use in another environment.

Each invocation runs one model in one process. The defaults use one compute thread, one model, a 3.5 GiB process memory check, and a 0.65 GiB host reserve. These checks are not hard memory isolation. Confirm local or allocated host capacity before changing model size or launching concurrent runs.

## Completed CPU conditions

Every configuration in `configs/` corresponds to a completed condition in the results index. No unexecuted expansion profile is included.

| Configuration prefix | Inputs and purpose | Calls per model/condition |
| --- | --- | ---: |
| `c3_*_native_chat`, `c3_*_raw_content` | 12 factual questions; candidate plus three samples | 48 |
| `c4_*_format_pilot`, `c4_*_no_example`, `c4_*_no_numbers`, `c4_*_no_example_no_numbers` | Four PVQ questions, two draws; initial exploratory controls | 8 |
| `c6_*_resource_probe` | Eight context questions; fit/token-budget check | 8 |
| `c6_*_development` | 400 balanced context questions | 400 |
| `repair_factual_*_original`, `repair_factual_*_repaired` | Six factual questions; paired wording diagnostic | 12 |
| `repair_value_*_released`, `repair_value_*_neutral_structured`, `repair_value_*_choice_only` | Eight PVQ questions, two draws; matched-seed validation | 16 |

`smol360` denotes SmolLM2-360M-Instruct; `qwen05` denotes Qwen2.5-0.5B-Instruct. The compact condition identifiers are preserved in configurations, recipes, and run hashes to make the historical results traceable.

```console
python -m trustbench.cpu --config configs/repair_factual_qwen05_repaired.json --out runs/factual_qwen_repaired
python -m trustbench.cpu --config configs/c6_smol360_development.json --out runs/context/smol360
```

The first use downloads the pinned checkpoint into the Hugging Face cache. `trust_remote_code` is disabled. On a cluster, run inside the approved persistent session or scheduler allocation and save logs on the remote host. Do not start a long foreground SSH job that depends on keeping the laptop connected.

To resume an interrupted, unchanged run:

```console
python -m trustbench.cpu --config configs/c6_smol360_development.json --out runs/context/smol360 --resume
```

Changing the configuration, data, any source module, or the recorded inference environment requires a new output directory. Do not edit a running package. A complete response list is materialized after all expected draws finish; the journal is the recovery source.

## Analysis

Inspect format validity under the contract actually requested:

```console
python -m trustbench.format_eval --run runs/value_choice --data data/cpu_measurement_repair_20261023/pvq_choice_only_validation.jsonl --contract choice_only --out reviews/value_choice_format.json
```

This example assumes a completed run created from `repair_value_qwen05_choice_only.json`. Format analysis does not assign reasoning quality or moral correctness.

For context labels, use the [annotation protocol](annotation.md). It documents packet generation, actual reviewer labels, and passage-group uncertainty. No automated semantic metrics should be reported as completed human grading.

Audit complete CPU journals against their frozen data:

```console
python -m trustbench.audit --runs runs/context --workspace . --out reviews/context_integrity.json
```

The audit checks response/draw identities, data hashes, seeds, grouping, saved candidates and samples, and raw likelihood accounting. To verify historical code, additionally supply `--code-root` pointing to the exact original source directory. The published run index records the original eight source hashes; the reorganized package deliberately has a different fingerprint. It cannot resume or retrospectively substitute for the original source.

Regenerate the published descriptive report and run the synthetic calibration diagnostic:

```console
python -m trustbench.report
python -m trustbench.calibration_diagnostic --trials 10000 --seed 679 --out reviews/calibration_synthetic.json
```

The second command performs 180,000 synthetic replications and may take appreciable CPU time. It is independent of model inference. The stored diagnostic is already complete; rerun it only when checking the implementation or a changed environment.

## Factual calibration and evaluation

The CLI also provides `calibrate` and `evaluate` for completed runs with reviewed boolean correctness labels and a frozen score protocol. Preserve raw results and write adjudicated records to a separate file beside their run metadata. The runner's automatic string-match `correct` field alone is insufficient to establish the main study's factual labels.

```console
trustbench calibrate --records runs/calibration/adjudicated.jsonl --out reviews/policy.json --alpha 0.05 --delta 0.01
trustbench evaluate --records runs/test/adjudicated.jsonl --task factual --calibration reviews/policy.json --bootstrap 2000 --out reviews/factual_metrics.json
```

These paths are an interface example, not existing completed calibration/test results. Calibration refuses failed or ungraded candidates and repeated groups unless the explicitly labeled published-reproduction mode is requested. Evaluation rejects calibration/test group overlap and a changed score/generator protocol. Statistical assumptions still require scientific validation.
