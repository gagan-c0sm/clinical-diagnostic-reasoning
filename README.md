# Local Clinical Diagnostic Reasoning

The aim is practical diagnostic performance on an RTX 4060 Laptop (8 GB), with
all model inference on-device. Extensive research ablations are out of scope.
This is a development prototype; output correctness has not been established.

## Installed runtime

The portable CUDA llama.cpp build is in `tools/llama.cpp`. Official
Qwen3-4B and Qwen3-8B Q4_K_M models are in `data/models`. No Ollama installation, global Python
packages, or cloud inference are used. Python scripts use the project `venv`.
The runtime is a native executable inside the project, outside the Python venv.
The NVIDIA driver remains a system dependency.

Pinned source revisions and SHA-256 hashes are recorded in
`tools/llama.cpp/manifest-4B.json` and `manifest-8B.json`; `manifest.json`
identifies the last launched model. `scripts/setup_local_runtime.py --model-size
4B` (or `8B`) retrieves and verifies these assets. Archives are retained in
`data/downloads`. Current disk usage: runtime 1.15 GiB, 4B weights 2.33 GiB,
8B weights 4.68 GiB, retained download archives 0.62 GiB.

## Run

From this project in PowerShell:

```powershell
.\scripts\start_local_model.ps1
.\venv\Scripts\python.exe -B scripts\run_pipeline.py --limit 5
```

To change models, stop the current project server first, then launch with
`-ModelSize 4B` or `-ModelSize 8B`. Both have been loaded on the GPU at an
8,192-token context. Observed total GPU memory during inference was about
4.0 GiB for 4B and 6.0 GiB for 8B; these are snapshots, not guaranteed peaks.

Wait for `listening on http://127.0.0.1:8081` in
`data/runtime/server.stderr.log` before running inference. The launcher uses an
8,192-token context, one inference slot, and GPU offload. It launches a hidden
process with local logs. It does not configure Windows startup or alter PATH.
Do not launch a second copy on the same port. Stop the specific llama-server
process with `.\scripts\stop_local_model.ps1` when finished to free GPU memory.

The Python HTTP client is fixed to IPv4 loopback, ignores proxy environment
variables and rejects redirects. The API's OpenAI-compatible format does not
contact OpenAI. Do not expose this server's port to the network.

Results are saved under `data/runs/<timestamp>/`: per-note predictions, errors,
source-grounding warnings, stage timing, token usage, and a compact summary.
Runs use development data by default. `--split test` is reserved for final
assessment, after choices are settled. There is no automatic retry with lost
context or truncated output. Such failures are recorded as errors.

## Data and schema decisions

- `data/splits_v2` is authoritative: 354 train / 77 dev / 80 test annotations,
  grouped by patient, with 335 KG-supported training records. The old
  `data/splits` is historical and must not be used for training/evaluation.
- A `sample_id` is the annotation's relative path under `samples/Finished`.
  There are 511 unique annotations but only 499 distinct original note IDs.
  Ambiguous lookups by original note ID now fail instead of returning an
  arbitrary annotation. Original note IDs and patient IDs are preserved.
- Canonical output is a flat node dictionary with `content`, `type`, `parent`,
  `children`, and `source_field` for inputs. The current/final diagnosis is the
  root; supporting diagnosis steps, rationales, and observations are beneath it.
  `src/pipeline/schema.py` validates this and exports nested source-style JSON.
- Original source annotations are never rewritten. Two have observations
  directly under diagnosis nodes, without a Cause node. They remain available
  as observations but cannot provide a complete observation/rationale triple.
- Explicit label aliases cover STEMI/STEMI-ACS, COPD Asthma/Asthma-COPD, and
  singular/plural pituitary adenoma labels. Thyroid Nodules is a valid internal
  KG milestone, not a leaf. Severe Asthma Exacerbation remains an unresolved
  distinction from Severe Asthma; we do not silently equate these labels.
- Gastritis has no supplied KG. Classification includes it, but KG reasoning
  returns `unsupported_kg` with no invented tree or diagnosis. Coverage is
  visible in results. A source-backed replacement can be added later.

## Pipeline behavior

1. Classify the admission's primary disease category from the note alone.
2. Select numbered source passages, including negative evidence, per input
   field. Text is copied directly from the source; invalid IDs are rejected.
   This replaces free-form quote generation, which produced too many invented
   or paraphrased quotes in the initial local smoke run.
3. Evaluate KG milestones with the retained observations, permitting STOP when
   evidence is insufficient. Assemble and validate the resulting tree in code.

The generation interface accepts input fields and KGs, not ground-truth labels.
Ground truth is attached by the runner only after generation for diagnostics.
Verbatim grounding checks do not establish that the medical rationale is sound.
The `complete` status means a valid tree reached a KG leaf; it does not certify
that the predicted diagnosis or explanation is correct.

The initial evaluator reports category accuracy, conservative alias-normalized
diagnosis accuracy, exact-span observation precision/recall, coverage, and time.
It is not the paper's semantic evaluator. Equivalent observation boundaries can
score poorly on exact-span metrics. Small development runs are debugging
evidence, not dependable estimates of final accuracy.

## Status and next decisions

D3 completed a 200-update pilot. Its adapter regressed on the matched development
check (7/12 diagnoses versus base 9/12), so further generic training was deferred.
D4 finished by early stopping at step 550; its best validation adapter is step
400 (loss 0.5670737710), exported as data/models/d4_best.gguf. Minimum-loss
adapters are preserved separately from the latest resumable checkpoint.

See [D4_RESULTS.md](D4_RESULTS.md) for the matched deployment comparison and
selected model. The later [behavior audit](MODEL_BEHAVIOR_REPORT.md) corrects two
reference-label mismatches: annotation-root diagnosis matches are base 15/25
and D4 17/25, versus the earlier metadata-based 15/25 tie. D4 is the stronger
provisional diagnosis-and-speed candidate on that cohort; the audit leaves the
configured base default unchanged. data/models/selected_model.json records the default used by the
manual case command. Training, failed sessions and assessments remain charged
in data/training/budget.json against the user's four-hour ceiling. No additional
training is planned, and final-test model inference has not run.

[MANUAL_TESTING.md](MANUAL_TESTING.md) explains JSON case input and standalone
HTML evidence/reasoning reports. For the supplied development example, run:

```powershell
.\scripts\start_local_model.ps1 -ModelSize 4B -AdapterPath data/models/d4_best.gguf
.\venv\Scripts\python.exe -B scripts\test_case.py examples/direct_dev_case.json
```

D1/D2 data loading, supplementary acquisition and formatting are complete.
D5 runs the local multi-stage pipeline. [D6 evaluation](EVALUATION.md) now
adds offline biomedical semantic matching for evidence and rationale steps,
one-to-one matching, milestone scoring, and auditable case-level results.
The threshold remains uncalibrated; the optional LLM judge is deferred. D7 now has manual JSON input and a readable HTML report; the original
single-pass model and interactive browser form remain unimplemented. D8 has a
practical development results report. Additional research comparison arms are
outside the agreed scope. D9's consolidated dataset-use documentation remains
pending; source data and generated case reports must stay within the applicable
MIMIC data-use conditions.

The optional shortlist router underperformed the default single router on the
matched development check. Label coverage and its limitations are recorded in
[DIAGNOSIS_COVERAGE.md](DIAGNOSIS_COVERAGE.md); training-label counts do not reveal
pretraining exposure. Gastritis has no supplied KG. A structurally valid,
source-grounded tree can still contain the wrong diagnosis or medical rationale.

Verification:

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
