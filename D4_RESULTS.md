# D4 outcome and practical deployment decision

**October 29 reference audit correction:** The earlier diagnosis table below
uses metadata PDD labels. Two cases have more specific final diagnoses in their
annotation trees (Moderate COPD and HFmrEF), both matched by D4. Against those
annotated final diagnoses the result is **base 15/25 versus D4 17/25**; incorrect
completed outputs are **base 6 versus D4 7**. The historical tie-based selection
therefore understates D4's gain. See MODEL_BEHAVIOR_REPORT.md for the full
decomposition, limitations and revised interpretation. Historical run files
are preserved; the deployed default has not been switched by this audit.

October 29, 2026. Keep the Qwen3-4B Q4_K_M base model with the existing
single-category pipeline as the default. D4 is saved and available explicitly,
but did not improve diagnosis matches on the completed matched development
comparison. This is a development decision, not a final accuracy estimate.

## Training outcome

D4 stopped normally at update 550 after three consecutive validation checks
failed to improve. The requested 600 updates were a ceiling. It consumed 1,100
task examples; it did not complete an epoch over the 1,200-example task set.
The minimum validation loss belongs to step 400, not the latest checkpoint.

| Update | Token-weighted validation loss |
|---|---:|
| Initial | 2.936171 |
| 50 | 0.906158 |
| 200 | 0.588523 |
| 400 — best | 0.567074 |
| 450 | 0.574167 |
| 500 | 0.579905 |
| 550 — stop | 0.568369 |

Validation monitored 30 fixed task instances, not the full 240-instance
development task export. These losses measure target-token prediction, not
diagnosis accuracy or explanation correctness.

The best adapter and metadata are preserved under
data/training/d4_session_600/best/step-000400/adapter and best.json. The regular
checkpoint-000400 includes optimizer/resume state; checkpoint-000550 is the
latest resume point. data/models/d4_best.gguf is the converted runtime adapter.
Minimum-loss saving updates on every strictly lower observed validation loss,
separately from the early-stopping improvement threshold.

The earlier CUDA OOM during update 85 was recovered from checkpoint 50.
Completion-only vocabulary logits retained all input tokens while reducing
memory. Full-loss equivalence and gradient tests and the failing-example replay
passed. Recovered training peaked at 5,665 MiB tensor allocation. Failed work
remains charged to the time budget.

## Deployment comparison

Both models used the same 25 development cases, covering all 24 development
categories and all three development cases whose labels are absent from project
training. Migraine is absent from development. Input, split, pipeline source,
prompts, GGUF base, routing and schemas were matched. Adapter scale was explicitly
verified at zero for base and one for D4. Selection revalidates tree structure
and verbatim source grounding for every nonempty output tree.

| Measure | Base | D4 step 400 |
|---|---:|---:|
| Diagnosis matches | 15/25 (60%) | 15/25 (60%) |
| Category matches | 17/25 (68%) | 19/25 (76%) |
| Mean inference time per case | 10.06 s | 6.88 s |
| Complete leaf outputs | 21 | 24 |
| Abstentions | 4 | 0 |
| Unsupported KG outputs | 0 | 1 |
| Incorrect complete leaf outputs | 6 | 9 |
| Train-present label diagnosis matches | 13/22 | 13/22 |
| Train-absent label diagnosis matches | 2/3 | 2/3 |

D4 fixed the bacterial-pneumonia example that base abstained on. It regressed
on the asthma/COPD example: base matched Asthma-COPD while D4 chose Moderate
COPD. The other 23 cases retained their correctness outcome. D4's faster runtime
and improved category matching are useful observations, but its higher completion
rate also included more incorrect diagnoses. More completed trees does not
establish better reasoning. Timing is sequential local measurement, without
repeated latency trials.

The earlier matched 12-case check was base 9/12 versus D4 8/12. The broader
check supersedes that cohort for selection; the cohorts overlap and must not be
added as independent evidence. The rule retains base on a diagnosis-count tie.
Selection is recorded in data/models/selected_model.json and can be reproduced
with scripts/select_deployment_model.py from the completed run files.

Evidence: data/runs/broad_base_dev and data/runs/broad_d4_dev contain predictions,
summaries and adapter configuration. The interrupted D4 attempt is preserved in
data/runs/broad_d4_interrupted and excluded from model scoring. Its connection
errors were infrastructure failures. The completed base run was retained;
only the unfinished D4 arm was restarted.

## Budget and usable case workflow

The shared ledger charges **151.48 minutes (2 h 31 min)** against the four-hour
ceiling, leaving 88.52 minutes. This includes a conservative historical charge
of 40 minutes, 80.96 minutes of D4 sessions (including failure), and 30.52 minutes
of assessment. The interrupted comparison was charged its entire enforced
20-minute allowance; idle time after process exit was excluded. The balance is
headroom, not a reason to spend it on more training.

Manual case testing is implemented in scripts/test_case.py. Supply the six
clinical input fields in JSON; the command uses the selected model and exports
the result plus a standalone HTML report. See MANUAL_TESTING.md. A live smoke
test using examples/direct_dev_case.json produced Paroxysmal Atrial Fibrillation
in 9.74 seconds, with 12 validated, source-grounded tree nodes. Its report is
data/manual_runs/verified_demo/report.html. This is a previously used development
example, not independent quality evidence. The local server was stopped afterward
to release GPU memory.

All 19 automated tests pass. No final-test model inference has run. Medical
rationale correctness has not been semantically or clinically validated.
The pipeline is limited to its supplied disease categories and KGs; unfamiliar
conditions may be misclassified, and Gastritis has no supplied KG. The current
manual report is the multi-stage pipeline, not a trained single-pass model.

Further practical work should inspect actual user case failures, especially
category routing, subtype criteria and abstention. Final held-out assessment
belongs after the deployment behavior is settled, without tuning on its results.
