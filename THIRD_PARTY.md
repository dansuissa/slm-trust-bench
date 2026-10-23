# External sources

The MIT license covers the code and documentation written for this repository. It does not relicense model weights, datasets, questionnaires, upstream implementations, or external prompt templates.

| Input | Source and pinned revision | Handling |
| --- | --- | --- |
| FactTest ParaRel files | [FactTest](https://github.com/fannie1208/FactTest/tree/2f7bb87530d8321423304330964b70e50b621307) | No repository license file was found at the audited revision. Downloaded locally; source files and derived question text are excluded from Git. |
| Questionnaire workbooks and prompt literals | [Value-Reasoning Reliability](https://github.com/Giovanna-SH/Value_Reasoning_Reliability/tree/0bbf0c5db2b7c8669a8a9d43cfd739a75d17bd84) | No repository license file was found at the audited revision. Downloaded locally; workbooks, questions, and released prompts are excluded from Git. |
| Judge prompt definitions | [AbstentionBench](https://github.com/facebookresearch/AbstentionBench/tree/e29184174c69bc95139b8c33dcca09aacab9442a) | Upstream [CC BY-NC 4.0 license](https://github.com/facebookresearch/AbstentionBench/blob/e29184174c69bc95139b8c33dcca09aacab9442a/LICENSE). Fetched locally with the license; no copied upstream implementation is distributed here. |
| SQuAD 2.0 validation data | [Dataset card](https://huggingface.co/datasets/rajpurkar/squad_v2), revision `3ffb306f725f7d2ce8394bc1873b24868140c412` | Card lists CC BY-SA 4.0. Fetched locally; passages and question text are excluded from Git. |
| CPU model checkpoints | [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) and [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache 2.0 model cards at the pinned revisions. Weights remain in a local cache. |

`manifest.json` records immutable URLs and SHA-256 hashes. `protocols.json` contains item identifiers and reconstruction recipes, without question text. The fetcher verifies bytes before saving and does not execute downloaded code. Public availability alone does not establish unrestricted reuse rights. Consult each source's terms before using or redistributing downloaded material, particularly questionnaire adaptations.

The published results contain aggregate measurements and provenance hashes. Raw prompts, generations, local annotation packets, SSH connection details, personal paths, and downloaded source archives are excluded.
