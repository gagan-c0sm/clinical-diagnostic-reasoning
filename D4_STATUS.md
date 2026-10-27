# Budgeted D4 preparation and completed session

October 29 update: D4 completed by early stopping at step 550. The best saved
validation checkpoint is step 400, loss 0.5670737710. Evaluations at 450, 500,
and 550 did not improve that minimum. The preserved adapter is under
data/training/d4_session_600/best/step-000400/adapter and its runtime export is
data/models/d4_best.gguf. No further training is planned. See D4_RESULTS.md
for the completed development comparison and model selection, and
MANUAL_TESTING.md for case input. The sections below record preparation and
recovery history; references to resumed/live training are historical.

October 28, 2026. The shared D3+D4 ceiling remains four hours, including a
conservative 40-minute charge for completed work and subsequent assessment.

## Recovery and best-adapter update

The first sustained run failed with CUDA OOM during update85, after completing
update84. Checkpoint50 survived (validation loss0.906158); updates51–84 were not
durable. Failure status is preserved as failure_step84.json, and original logs
remain available. The failed session's time remains charged to the budget.

The model now computes vocabulary logits only for completion positions, retaining
the complete clinical input. The failing1925-token example had only27 supervised
completion tokens. A unit test verifies the optimized causal loss and gradients
against full-sequence loss. Replaying the failed two-example update succeeded
at5374 MiB allocated /5918 MiB reserved. Cached CUDA memory is released after
each optimizer update. This mitigates the observed failure; it does not prove
that every future GPU allocation will succeed.

Training was resumed from checkpoint50. New logs are
data/training/logs/d4_recovered.stdout.log and d4_recovered.stderr.log.

Best-validation saving is explicit: best.json records the minimum observed
validation loss, step, regular resume checkpoint and preserved adapter path.
Adapters are copied under best/step-NNNNNN/adapter. A pointer is published only
after its complete copy exists. Better losses update it, including improvements
smaller than the early-stopping threshold; ties/worse losses retain the previous
best. Older best copies and regular checkpoints are preserved. Best selection
uses validation evaluations every50 updates and at session end, not noisy
individual training-batch losses. Post-training assessment reads this best
adapter manifest. Current progress remains in status.json.

## Decision after controlled D3 transfer check

The same GGUF base, pipeline, prompts, JSON constraints and 12 development cases
were run with the D3 LoRA scale explicitly set to zero and one. Base: 9/12
diagnosis matches. D3 pilot: 7/12, including one classification output-length
failure. These are small development checks, not established accuracy estimates.
The exported adapter changes quantization from NF4 training to a Q4_K_M base at
inference, so this assesses the deployment path rather than isolating training
effects. Files are under data/runs/d3_transfer_base_12 and d3_transfer_pilot_12.

Further D3 training is deferred. D4 starts from the original HF base with a new
LoRA adapter; it does not continue the regressing D3 adapter. The D3 checkpoint
is preserved. This departure from the original sequential curriculum follows
the observed deployment results and the time constraint.

## Diverse D3 subset, prepared but not trained

data/d3_curated has 1,600 new examples: 960 MedCaseReasoning and 640 Medical-O1,
excluding the 800 consumed pilot examples and all validation inputs. It contains
1,435 topic proxies, 1,055,959 tokens, and 837 explicit lexical matches to project
categories. Sampling covers source, topic, length and reasoning-skill proxies.
Proxy tags are not verified specialties or diagnoses. Approximate minhash
duplicate screening removed no additional records; paraphrases can remain.
The original supplementary validation split is retained unchanged.

## Short D4 targets

data/d4_compact contains 1,200 training and 240 validation instances. Prompts and
JSON schemas are captured from the existing pipeline. Targets are derived from
training annotations: category, numbered source-passage selection, and one KG
decision with annotated rationale/evidence. Setup nodes used while capturing a
branch prompt are not training targets. No synthetic negative diagnoses or STOP
labels are inferred merely from missing annotations. This leaves limited
supervision for abstention, which needs checking during evaluation.

| Training task | Instances | Distinct training annotations |
|---|---:|---:|
| Category | 302 | 302 |
| Source passages | 554 | 335 |
| KG decisions | 344 | 334 |

Combined: 350/354 training annotations, all 25 categories. All 335 KG-supported
training annotations have passage tasks. Four Gastritis annotations have no
fitting category instance and no KG; IDs are listed in the manifest. Fifty-two
category and 27 passage candidates exceeded 2,048 tokens and were excluded,
not truncated. There are no KG-dependent Gastritis tasks. All 77 development
annotations are represented across validation tasks. Final-test annotations
are not used for task preparation or inference.

Training has 1,258,218 tokens, mean length 1,048.5, median 983, p90 1,668, max
2,047. The dataset retains at least one fitting instance per available case/task
pair and caps each case at four tasks. This preserves coverage but is not a
uniform disease distribution; category counts are in manifest.json.

## GPU and integrity checks

The two longest tasks (2,047 and 2,044 tokens) passed NF4 QLoRA training and exact
adapter tensor/loss reload checks. Peak tensor allocation was 6,939 MiB; allocator
reservation reached 8,206 MiB under Windows. A subsequent session check also
showed high cache reservation. Live tensor memory fit, but Windows shared-memory
behavior can affect speed. Unused CUDA cache is released after validation.
Do not run the inference server alongside training.

All 15 tests pass, including cumulative budget charging, exclusion of pilot and
validation inputs from curation, D4 split membership, target integrity and length.

## Session and budget controls

The verified session data/training/d4_session_600 resumed from update 3. It allows
at most 600 updates at effective batch 2 (one pass through 1,200 tasks), learning
rate 5e-5, rank8 QLoRA, validation/checkpoints every50 updates, and early stopping
after three non-improving validation checks. Validation is fixed and balanced
across 30 tasks (10 per task type), not across all diagnoses.

The persistent data/training/budget.json ledger charges each load/train/validate/
checkpoint session across restarts. D4 has 100 minutes total, including its
three-update session check. SharedBudget serializes budgeted GPU jobs, refuses
exhausted allocations, and records active process identity. Training stops with
a three-minute checkpoint margin; a watchdog terminates a stalled job at its
deadline, preserving the last complete checkpoint. A crashed active lease must
be explicitly recovered after checking its process has exited; it is charged
conservatively, never silently reset.

scripts/continue_d4.py runs training, converts the best validation checkpoint,
then assesses the same 12-case pipeline cohort inside the remaining assessment
allowance. It records selection.json; a candidate is provisionally preferred
only if diagnosis matches exceed the 9/12 base result. Equal/worse results retain
the base. This development rule does not establish clinical reasoning quality
or final-test performance. Baseline weights are never overwritten.

Current progress: data/training/d4_session_600/status.json.
Logs: data/training/logs/d4_session_600.stdout.log and .stderr.log.
Training is not complete merely because the continuation process was launched.
