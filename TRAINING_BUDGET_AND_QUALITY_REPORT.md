# Four-hour training budget and current model quality

Prepared October 28, 2026. This is a report and proposed plan. No additional
training was started for this report. The 200-update pilot has completed.

## Findings and recommendation

Keep checkpoint-000200 as the current D3 candidate. Its validation loss improved
substantially, but diagnostic reasoning quality is not yet established. Do not
train a full supplementary epoch. Prefer a small, diverse continuation only
if it improves transfer, then prioritize D4 tasks aligned with the working
pipeline. Enforce a cumulative four-hour budget across both stages, including
the completed pilot, validation, checkpointing and short training trials.

## 1. What has actually run

The pilot completed 200 optimizer updates on 800 distinct supplementary training
examples: 360 MedCaseReasoning and 440 Medical-O1. It processed 510,166 input-plus-
completion tokens. It used one example per microbatch and four microbatches per
optimizer update. Approximate process wall time was 26.5 minutes, measured from
log-file creation to final write; precise per-update timestamps were not logged.

The distinction between examples and updates matters:

- 31,754 is the number of training examples.
- At gradient accumulation 4, one epoch is about 7,939 optimizer updates.
- At 8 optimizer updates/minute, that is about 16.5 hours before overhead.
- Treating all 31,754 examples as optimizer updates would instead imply 66 hours.

Your concern is correct: a full epoch is incompatible with the new budget.
The pilot consumed only 2.5% of the prepared supplementary training examples.

## 2. Validation results

Validation used the same 32 examples throughout: 16 per supplementary source.
It is a small monitoring subset of the 1,489-example validation split.

| Optimizer update | Token-weighted validation loss |
|---:|---:|
| 0 | 2.409445 |
| 50 | 1.522494 |
| 100 | 1.466264 |
| 150 | 1.447168 |
| 200 | **1.441871** |

The best saved checkpoint is
`data/training/d3_pilot_200/checkpoint-000200/adapter`.
Loss decreased approximately 40.2%. At update 200, source-specific losses were
1.491922 for Medical-O1 and 1.330638 for MedCaseReasoning. Both improved relative
to their initial monitoring losses. Marginal improvement diminished sharply:
the last 50 updates improved loss by only 0.0053.

Loss measures prediction of reference completion tokens under teacher forcing.
It is not diagnostic accuracy, explanation correctness, calibrated confidence,
or a percentage of correctly solved cases. It can improve through learning
response style and terminology as well as medical associations. The small,
repeated monitoring set does not establish broad generalization.

## 3. Can the best checkpoint produce good reasoning?

It can generate organized diagnostic explanations. Reliable medical reasoning
has not been demonstrated. A local qualitative check compared the base NF4
model and the pilot adapter with the same prompts, greedy decoding and a
256-new-token limit. There were four cases: two MedCaseReasoning validation
cases outside the monitoring subset and two known DiReCT development cases.
This was direct generation without KG routing, not the full pipeline benchmark.

| Reference | Base response | Pilot response |
|---|---|---|
| Small intestinal perforation | Acute/plastron appendicitis | Plastron appendicitis |
| Pilocytic astrocytoma | Broad tumor/meningitis differential | Intracranial meningioma |
| NSTEMI | Non-cardiac chest pain / other possibilities | Broad acute coronary syndrome; subtype omitted |
| COPD | COPD | Pulmonary hypertension |

The pilot's responses were shorter and completed within the token limit. All
four base responses hit the cap, so comparisons of completeness are biased by
that limit. The displayed pilot diagnoses did not reproduce any of the four
specific reference labels; the ACS response was related to the reference but
less specific. Four selected cases are not an accuracy estimate. Some cases
may also be difficult to resolve from the presented findings alone.

The pilot also put paraphrases in quotation marks: none of its three quoted
passages in the ACS response were exact source quotes. In the COPD response,
only one of six quoted spans was verbatim; that count includes explanatory
phrases, so it is a quotation-fidelity check rather than an observation metric.
This reinforces the value of deterministic source-passage selection in the
existing pipeline. Verbatim checks alone do not verify the medical inference.

The previous 9/12 pipeline result belongs to the unadapted model with KG-guided
generation. It must not be attributed to this pilot. The best-loss checkpoint
has not yet been evaluated on that same pipeline protocol. The final DiReCT
test set has not been used for model predictions.

Raw local examples: `data/audits/d3_generation_check.jsonl`.

## 4. Proposed four-hour envelope

Count the work already completed rather than starting a fresh four-hour clock.
Use the following conservative planning envelope:

| Allocation | Minutes | Scope |
|---|---:|---|
| Completed work charged to budget | 40 | 26.5-minute pilot plus allowance for earlier short training/resume trials |
| Optional additional D3 | **up to 60** | Curated examples, at most 400 additional optimizer updates |
| D4 specialization | **up to 100** | Short, pipeline-aligned supervised tasks |
| Shared reserve | **40** | Validation, checkpointing, load/reload, assessment and timing uncertainty |
| **Combined maximum** | **240** | Includes prior work and overhead |

The 40-minute completed-work charge is conservative bookkeeping, not a claim
that historical timing was measured precisely. Any extra verification/training
must consume the reserve. If D4 is slower than estimated, reduce updates; never
extend the four-hour limit. Unused D3 allocation can be retained or reassigned
to D4 within the same total ceiling. Spending the full allowance is optional.

A persistent shared ledger must track elapsed time across stages and resumed
processes. Separate per-process one-hour caps are insufficient because they
reset on resume. Check remaining time before optimizer work and reserve time
for the final checkpoint. Log elapsed seconds, processed tokens, examples and
optimizer updates. Calibrate D4 throughput from a short trial charged to reserve.

## 5. Curating D3 for useful exposure

Select at most 1,600 additional distinct examples, trained once. With effective
batch 4, that is 400 updates: approximately 50 minutes at the observed rate,
leaving ten minutes within the D3 allocation for overhead. This would provide
at most 2,400 distinct examples including the existing pilot, not repeated
epochs over a tiny random sample. A nested 800-example subset offers a cheaper
200-update option. Given diminishing returns, extra D3 is conditional rather
than mandatory; the current pilot may already be the better starting point for D4.

Proposed maximum source quotas: 960 MedCaseReasoning and 640 Medical-O1.
Prioritize patient-case diagnostic reasoning while retaining question-based
reasoning diversity. Reconstruct the consumed pilot IDs from the fixed shuffle
and exclude those 800 IDs from the new selection. Preserve validation/test
boundaries and screen near-duplicate questions, not just exact duplicates.

Within the source quotas:

1. Build topic groups from diagnosis text (MedCaseReasoning) and question text
   (Medical-O1). Add tags for the project's disease categories using KG names
   and aliases. These are proxy strata, not verified clinical specialty labels.
2. Allocate coverage across distinct topics and rare diagnoses rather than
   allowing common themes to dominate. Include relevant cases beyond the 25
   DiReCT categories so the model is not trained only on those labels.
3. Include varied reasoning tasks: differentials, discriminating tests,
   quantitative findings, negative evidence, comorbidity, and uncertainty.
   Validate inferred tags on a small sample rather than presenting them as gold.
4. Use length quantiles, with most examples in the 400–900-token range and a
   minority of longer complete cases. Do not discard the longest cases solely
   for speed or truncate final answers. Fit selection to a token budget as well
   as an example budget.
5. Prefer distinct lexical/semantic clusters; limit repeated variants of the
   same question. Apply quotas and a diversity score with a fixed seed, and
   export a coverage manifest so the selection is reviewable.

This is a practical coverage heuristic. No inexpensive selector can guarantee
that a record gives the model the most learning benefit. A continuation would
initialize from the current best adapter with a fresh, documented schedule;
the existing completed 200-step scheduler cannot simply be extended by changing
its configured step count.

## 6. D4 needs shorter targets before training

Measured full-tree examples, including chat formatting, clinical text, KG and
JSON target:

| Statistic | Tokens |
|---|---:|
| Examples | 335 |
| Minimum | 1,054 |
| Median | 3,033 |
| 90th percentile | 4,997 |
| Maximum | 8,506 |
| Fit within 2,048 | **58 / 335 (17.3%)** |

Training these full examples directly at 2,048 would omit most cases or cut
their reasoning trees. Neither is acceptable for broad coverage.

Recommended D4 supervision matches the existing three-stage pipeline:

- Category selection from clinical text.
- Selection of relevant numbered source passages for a field and category.
- One KG branch decision with its criterion, relevant evidence, rationale and
  an explicit insufficient-evidence option where annotations support it.

Assemble complete trees deterministically in code, as the pipeline already
does. This changes the training target from a long full-tree dump to bounded
reasoning decisions. It may improve this pipeline without teaching single-pass
full-tree generation; the fast full-tree demo remains a separate capability.

Use training annotations only: 354 are available for category supervision and
335 have supplied KGs for KG-dependent tasks. Preserve all available category
coverage, balance rare diagnosis labels, and cap per-case contributions so deep
trees do not dominate. Resolve or explicitly exclude label/KG mismatches.
Missing annotation evidence must not be relabeled as a clinical negative.
Gastritis still lacks KG-dependent supervision.

Provisional cap: about 1,200 compact training instances, effective batch 2 and
at most 600 updates. At a conservative eight updates/minute this is 75 minutes,
leaving 25 minutes in the D4 allocation. Actual per-update time will differ
because both sequence lengths and accumulation change. Reprofile lengths and
measure throughput before fixing the final update count. If needed, reduce
examples/epochs while retaining rare-category coverage. Do not exceed the cap
to chase lower loss.

## 7. Decision gates

Before more generic training, prefer a controlled development comparison of
the current adapter against the base under identical pipeline conditions.
Review correct diagnoses, missed evidence, unsupported branch decisions and
quotation fidelity. A fixed validation-loss decline alone is insufficient.
Use a broader source-balanced validation sample for selection checks within
reserve; continue to keep final-test predictions out of tuning.

If transfer is neutral or worse, skip additional D3 and prioritize compact D4.
If D3 helps, use the nested curated subset with early stopping. Select the
checkpoint that improves the task while remaining within the shared clock.
No additional training has been launched pending review of this report.
