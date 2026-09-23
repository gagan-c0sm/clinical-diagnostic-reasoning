# Diagnosis coverage audit

Present/absent refers to project training data, not pretraining. No project fine-tuning has run.

There are 55 diagnosis labels; 51 occur in the training split and 48 in the usable DiReCT SFT export.

| Split | Annotations | Categories | Labels | Cases with label absent from train |
|---|---:|---:|---:|---:|
| train | 354 | 25 | 51 | 0 |
| dev | 77 | 24 | 35 | 3 |
| test | 80 | 23 | 34 | 3 |

## Labels absent from training

| Diagnosis | Train | Dev | Test | KG |
|---|---:|---:|---:|---|
| Allergic Asthma | 0 | 0 | 1 | leaf |
| Hypertrophic Cardiomyopathy | 0 | 1 | 1 | leaf |
| Massive PE | 0 | 1 | 1 | leaf |
| Thyroiditis | 0 | 1 | 0 | leaf |

## Existing small development runs

| Run | Present-label correct / n | Absent-label correct / n |
|---|---:|---:|
| dev_4b_corrected_routing_10 | 7/9 | 1/1 |
| dev_8b_corrected_routing_10 | 4/9 | 1/1 |

The absent-label case in each previous ten-case batch was Hypertrophic Cardiomyopathy. One success is not a dependable estimate of generalization.

Final-test labels were inspected for coverage only; no final-test predictions were generated. Splits remain frozen. See the CSV for all labels and KG gaps.

Three Gastritis labels are present in the training split but absent from usable SFT because no KG is supplied. This is tracked separately from labels absent from training entirely.

No project-trained model exists yet, so these strata describe data coverage rather than a learned seen/unseen effect.
