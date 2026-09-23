# Category routing development check

Same 4B model and development cohort, including all three diagnosis labels absent from project training. No fine-tuning has run.

| Routing | Category correct | Diagnosis correct | Present-label diagnosis | Absent-label diagnosis | Mean seconds |
|---|---:|---:|---:|---:|---:|
| single | 10/12 | 9/12 | 7/9 | 2/3 | 11.0 |
| shortlist | 7/12 | 6/12 | 4/9 | 2/3 | 12.0 |

Shortlist fixed 0 baseline diagnosis errors and lost 3 baseline successes.

Expected-category coverage in the shortlist: 91.7%.

## Changed diagnoses

| Expected diagnosis | Single result | Shortlist result |
|---|---|---|
| NSTEMI | HFpEF (complete) | None (abstained) |
| Type A Aortic Dissection | Type A Aortic Dissection (complete) | None (abstained) |
| Paroxysmal Atrial Fibrillation | Paroxysmal Atrial Fibrillation (complete) | None (abstained) |
| COPD | HFpEF (complete) | None (abstained) |
| Type II Diabetes | Type II Diabetes (complete) | None (abstained) |

This is a small development comparison. Labels absent from project training may still be familiar from pretraining; their KG criteria are supplied. Final-test predictions were not generated.
