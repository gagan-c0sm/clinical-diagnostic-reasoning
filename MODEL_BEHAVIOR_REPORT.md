# Base versus D4: observed behavior in the broad development comparison

October 29, 2026. This audit uses the saved 25-case outputs; it adds no model
generation or training. It covers 24 development categories, with two PE cases,
and all three development labels absent from the project training split.
The cohort is small, deliberately category-balanced, and was used for model
development. Results describe these cases, not clinical deployment accuracy.

## A scoring correction changes the headline

The previous answer reported a 15/25 tie against metadata PDD labels. Inspection
of the actual annotation trees found two non-equivalent reference labels:

| Case | Metadata PDD | Annotated final diagnosis | Base output | D4 output |
|---|---|---|---|---|
| COPD, 15166831-DS-16 | COPD | Moderate COPD | HFpEF | Moderate COPD |
| Heart failure, 11465548-DS-9 | Heart Failure | HFmrEF | HFrEF | HFmrEF |

D4 matched the final annotated diagnosis in both. These are not new aliases:
the broad condition and subtype remain distinct. The correct response is to
report both endpoints explicitly, rather than silently treating every subtype
as equivalent to a category. Earlier summaries and selection.json used metadata
labels; their original files are preserved. The evaluator now reports the
annotation-root endpoint alongside metadata accuracy, with a regression test.

| Measure | Base | D4 step 400 |
|---|---:|---:|
| Metadata-PDD matches, original endpoint | 15/25 (60%) | 15/25 (60%) |
| Annotated-final-diagnosis matches | 15/25 (60%) | **17/25 (68%)** |
| Disease-category matches | 17/25 (68%) | **19/25 (76%)** |
| Completed leaf outputs | 21 | 24 |
| Correct among completed outputs, annotated final diagnosis | 15/21 (71.4%) | 17/24 (70.8%) |
| Incorrect completed outputs, corrected endpoint | 6 | 7 |
| Abstentions | 4 | 0 |
| Unsupported-KG outputs | 0 | 1 |
| Mean inference time | 10.06 s | **6.88 s** |

Both models matched 14 final diagnoses. D4 alone matched three: bacterial
pneumonia, Moderate COPD and HFmrEF. Base alone matched Asthma-COPD. Both missed
seven. D4 therefore has a modest diagnosis-and-speed advantage on this cohort;
the earlier conclusion that there was no diagnosis gain needs revision.
The configured default remains base during this audit; no model was switched.

## 1. Routing: D4 improves some categories but remains vulnerable to overlap

Base had eight category failures: five committed to another category and three
returned Uncertain. D4 had six, all committed to another category. Compared
with base, D4 fixed routing for COPD, pneumonia and low-risk PE, while losing
Asthma-COPD by routing it into COPD. Other category failures persisted.

There are two distinct recurring problems:

- **Target-focus ambiguity.** The hyperlipidemia note prominently describes
  early satiety and abdominal discomfort but the annotation targets the lipid
  finding. Base noticed triglycerides in its summary yet returned Uncertain;
  D4 chose Gastritis from the presenting symptoms. The hypertension note
  prominently describes abdominal pain and chronic pancreatitis, while the
  annotation targets blood pressure. Base routed to UGIB and then stopped;
  D4 routed to GERD and completed that diagnosis. These demonstrate a mismatch
  between selecting the prominent presenting problem and recovering the
  dataset's annotated target; they are not evidence that either model has
  identified the sole possible disease in the note.
- **Overlapping category boundaries.** Both selected UGIB for the duodenal-ulcer
  case, recognizing the bleeding syndrome but missing the annotated etiology.
  Both selected heart failure for the NSTEMI case. D4 selected COPD for the
  Asthma-COPD overlap case despite the bronchodilator-reversibility evidence
  used successfully by base. Once routing is fixed, the downstream loop sees
  that category's KG and cannot recover an alternative category.

Gastritis has no supplied KG. Both models misrouted the actual gastritis case;
even correct routing could not produce a supported Gastritis reasoning tree
with the current assets. D4 also hit this missing-KG path after wrongly routing
the hyperlipidemia case to Gastritis. This is an implementation/coverage limit
in addition to model routing behavior.

## 2. Evidence extraction: D4 is more selective, including with routing held equal

All extracted observation passages were verbatim grounded in the source, and
neither model produced a structurally invalid nonempty tree. This follows from
passage-ID selection and deterministic assembly; it does not prove that the
passages are relevant or that the inference from them is correct.

| Semantic evidence measure | Base | D4 |
|---|---:|---:|
| Extracted observations | 400 | 370 |
| Matched reference observations | 144 | 166 |
| Precision | 0.360 | **0.449** |
| Recall | 0.735 | **0.847** |
| F1 | 0.483 | **0.587** |

Some of this improvement could come from routing into the correct KG more
often. On the **16 identical cases where both routed correctly**, base selected
300 observations and matched 108; D4 selected 234 and matched 109. Evidence F1
was 0.509 versus 0.609. On that controlled subset, the clearest improvement is
selectivity: D4 retained about the same matched evidence with fewer extra
passages. The matcher uses the same field and one-to-one correspondence; its
0.80 embedding threshold remains uncalibrated.
An unmatched passage is not automatically clinically irrelevant: annotations
are selective, and span boundaries and similarity thresholds affect matching.

Extraction and use are different. D4 can extract relevant evidence and then
cite an insufficient passage for the final branch. For low-risk PE it chose
Massive PE using only the source phrase “Chest pain” and the rationale “Chest
pain is a common symptom of PE.” That supports neither the severity choice nor
the supplied Massive-PE KG requirement of hemodynamic instability.

## 3. Reasoning style: D4 learned terse, single-anchor responses

| Output behavior | Base | D4 |
|---|---:|---:|
| Mean words in category explanation | 60.6 | 9.9 |
| Mean words in reasoning rationale | 49.0 | 11.3 |
| Mean evidence IDs per reasoning decision | 6.81 | 1.00 |
| Decisions citing exactly one passage | 8/53 | **59/59** |
| STOP decisions | 1 | 0 |

All 25 D4 classification summaries begin with “Annotated findings:”, the prefix
used in the training targets. This is direct evidence of learning the target
format, not of access to ground-truth annotations at inference: the pipeline
still receives the note and KGs only.

The compact D4 training set contains **344 reasoning targets: 342 with one
evidence ID, one with two, one with three, and zero STOP targets**. Their mean
rationale length is 13.4 words. Positive, single-chain targets were selected to
fit the hardware/time budget. D4's observed output style closely follows that
supervision. A plausible interpretation is that training improved concise
annotation-style responses more strongly than multi-evidence comparison,
contradiction handling or abstention. This is a supported inference, not an
isolated causal experiment.

The base model integrates more evidence but can rationalize a wrong branch:

- On heart failure it quoted LVEF 45–50%, then selected HFrEF while stating that
  its supplied criterion was LVEF ≤40%. Its own text exposes the contradiction.
  D4 selected the reference HFmrEF branch from the numerical finding.
- In the COPD case, base acknowledged obstructive spirometry but followed the
  heart-failure KG to HFpEF. It variously described proBNP 124 as supporting
  heart failure and as normal. More quoted evidence did not prevent anchoring
  on the wrong category.

D4's sparse pattern creates a different failure: repeating a broadly relevant
fact where the branch needs a distinguishing criterion. In the NSTEMI case,
its wrong heart-failure path reuses ventricular hypertrophy to advance through
multiple milestones. In low-risk PE, the final severity rationale merely
repeats a general PE symptom. Neither pattern tests alternatives adequately.

## 4. Explanation matching: D4's cleaner evidence did not yield stronger explanations

At the fixed, uncalibrated threshold, base matched 33/52 generated rationale
steps; D4 matched 35/59. Explanation precision was 0.635 versus 0.593, and recall
against 216 reference rationale steps was 0.153 versus 0.162. D4 generated more
steps because it completed more cases; the total match count alone is not a
quality improvement.

On the 16 cases where **both categories were correct**, each generated 37
rationale steps. Base matched 31 (precision 0.838), D4 matched 28 (0.757).
This is consistent with weaker annotation agreement in D4's compressed
explanations even after reducing routing confounding. It is not a clinical
correctness judgment: embedding similarity can reward word overlap and miss
negation or numeric errors. The base HFrEF contradiction is an example of why
long, similar-sounding explanations still need inspection.

The low explanation recall also reflects a representation difference. The
pipeline emits one rationale per chosen milestone; reference trees can contain
multiple supporting rationale nodes for the same milestone. It should not be
read as a calibrated percentage of “reasoning ability.”

## 5. Abstention and commitment

Base returned Uncertain during classification on hyperlipidemia, pneumonia
and low-risk PE. It also stopped during reasoning on the hypertension note
after rejecting its own UGIB route. That last STOP is a useful behavior: the
model's explanation recognized that the chosen category lacked support.

D4 converted those four base abstentions into one correct pneumonia diagnosis,
two incorrect completed diagnoses (GERD and Massive PE), and one unsupported-KG
output. It never chose Uncertain or STOP. This demonstrates greater commitment,
not measured confidence calibration. The absence of STOP training targets is
a concrete supervision gap.

After correcting the endpoint, completed-output correctness is almost equal
(71.4% base versus 70.8% D4). It would therefore be misleading to keep the earlier
claim of nine wrong completed D4 diagnoses: the corrected number is **seven**.
D4 answers more cases, gains two correct final diagnoses overall, and makes one
additional incorrect completed diagnosis overall.

## 6. Shared severity and temporal failures

Both predicted Submassive PE for the annotated Massive-PE case. The source
contains initial hypotension to 76/52, while later passages describe stabilization
and discharge. Base explicitly called the patient stable despite citing a
passage containing the earlier hypotension. D4 used RV enlargement alone for
the subtype and did not integrate the hypotension into that final decision.
The annotation's final rationale relies on the earlier hypotension.

This is a concrete example of failure to align the diagnostic decision with
the relevant point in the clinical timeline. It does not establish a general
temporal-reasoning failure rate, but it explains why reaching the correct PE
category was insufficient. Together with low-risk PE, severity differentiation
is an unresolved problem for both models on this cohort.

## 7. Why D4 is faster

Base generated 11,893 completion tokens across 205 calls; D4 generated 5,451
across 223 calls. D4 performed more calls but wrote about **54% fewer output
tokens**, including 2,597 reasoning tokens versus base's 7,173. Aggregate
reasoning-call time fell from 140.6 to 73.9 seconds. Mean case time fell by 31.6%.
Shorter outputs and fewer evidence IDs are consistent with this speed gain.
These are sequential run measurements, not repeated latency trials; cache and
runtime effects have not been isolated.

## Every case, with the corrected endpoint

✓ means an alias-normalized match to the annotation tree's final diagnosis.
“No answer” is distinct from an incorrect completed answer.

| # | Annotated final diagnosis | Base | D4 |
|---:|---|---|---|
| 1 | NSTEMI | HFpEF ✗ | HFpEF ✗ |
| 2 | Congenital Adrenal Hyperplasia | ✓ | ✓ |
| 3 | Alzheimer | ✓ | ✓ |
| 4 | Type A Aortic Dissection | ✓ | ✓ |
| 5 | Asthma-COPD | ✓ | Moderate COPD ✗ |
| 6 | Paroxysmal Atrial Fibrillation | ✓ | ✓ |
| 7 | Moderate COPD | HFpEF ✗ | ✓ |
| 8 | Hypertrophic Cardiomyopathy | ✓ | ✓ |
| 9 | Type II Diabetes | ✓ | ✓ |
| 10 | Epilepsy | ✓ | ✓ |
| 11 | Acute Gastritis | UGIB ✗ | GERD ✗ |
| 12 | GERD | ✓ | ✓ |
| 13 | HFmrEF | HFrEF ✗ | ✓ |
| 14 | Hyperlipidemia | No answer: Uncertain | No answer: routed to missing Gastritis KG |
| 15 | Hypertension | No answer: UGIB route then STOP | GERD ✗ |
| 16 | Relapsing-Remitting Multiple Sclerosis | ✓ | ✓ |
| 17 | Duodenal Ulcers | UGIB ✗ | UGIB ✗ |
| 18 | Pituitary Macroadenoma | ✓ | ✓ |
| 19 | Bacterial Pneumonia | No answer: Uncertain | ✓ |
| 20 | Low-risk PE | No answer: Uncertain | Massive PE ✗ |
| 21 | Hemorrhagic Stroke | ✓ | ✓ |
| 22 | Thyroiditis | ✓ | ✓ |
| 23 | Tuberculosis | ✓ | ✓ |
| 24 | Upper Gastrointestinal Bleeding | ✓ | ✓ |
| 25 | Massive PE | Submassive PE ✗ | Submassive PE ✗ |

## Practical interpretation

For the goal of diagnosis performance and speed on the available hardware,
**D4 is now the stronger provisional candidate on these cases**: 17 versus 15
annotated final diagnoses, better evidence selection, and lower latency. It
does not demonstrate a broadly reliable diagnostic reasoner. Its main weakness
is choosing a specific branch from one nonspecific fact, with no learned STOP
behavior. Base's advantages are broader evidence integration and an observed
ability to reject a bad route; its weaknesses include verbose rationalization,
unnecessary abstention, and explicit criterion contradictions.

The most useful future adaptation would address multi-evidence, contrasting
branches, explicit STOP examples and temporal criteria. More repetitions of
the current positive single-evidence targets would not directly address those
gaps. Correct reference-target handling should precede any next model selection.
This audit itself changes no training, deployment settings or final-test data.

Reproduce the decomposition with scripts/analyze_model_behavior.py.
data/evaluation/behavior_audit/audit.json contains all case IDs, both reference
labels, stage counts/times, training-target counts, actual reasoning traces and
the final-step cited passages. Source predictions remain in
data/runs/broad_base_dev and data/runs/broad_d4_dev. Semantic matching details
and limitations are in EVALUATION.md.
