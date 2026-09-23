# D6: local evaluation of saved predictions

The evaluator scores existing pipeline outputs against DiReCT annotations.
It does not generate predictions, train a model, change the selected model, or
send clinical text to a service. CPU is the default, so the inference server
can remain stopped. Torch and Transformers use the existing project venv.

## Run

The pinned biomedical encoder is downloaded once into data/models:

```powershell
.\venv\Scripts\python.exe -B scripts/setup_evaluation_model.py
```

Score completed development runs:

```powershell
.\venv\Scripts\python.exe -B scripts/evaluate_predictions.py data/runs/broad_base_dev data/runs/broad_d4_dev --output data/evaluation/my_dev_run
```

Use a new output directory for every run. Existing results are never overwritten.
Default threshold is 0.80; --threshold changes it explicitly. Default split is
dev; scoring saved test predictions requires --split test. Predictions must
belong to the declared frozen split, have unique IDs, and agree with the run's
record count. Reference labels are loaded from the dataset, not trusted from
the prediction file. This command never runs final-test inference.

## Metric definitions

| Measure | Definition |
|---|---|
| Metadata PDD accuracy | Fraction with an alias-normalized exact metadata PDD match; retained for historical comparison |
| Annotated final-diagnosis accuracy | Match to the actual annotation-tree root, reported separately because some metadata PDDs are broader than the annotation |
| Category accuracy | Fraction with the annotated category |
| Observation precision/recall/F1 | Maximum one-to-one evidence matching; both exact and embedding variants |
| Explanation precision (Exp_all proxy) | Matched predicted rationale steps / predicted steps |
| Explanation recall (Exp_com proxy) | Matched reference rationale steps / reference steps |
| Diagnostic milestone precision/recall/F1 | Normalized diagnosis-label set overlap |
| Structural and grounding diagnostics | Tree errors, counts of verbatim-grounded extracted observations and output statuses |

Evidence matching requires the same input field. Predicted text must occur
verbatim in that field. Semantic matching uses cosine similarity at or above
the configured threshold. Duplicate normalized evidence within a field is
collapsed. A reference passage can match at most one predicted passage.

An explanation step is a Cause node with its parent diagnosis and attached
observations. It matches only when milestone labels agree, rationale similarity
passes the threshold, and at least one same-field evidence pair also passes.
Each reference/predicted step participates in at most one match. This prevents
repeated evidence or a generic rationale under the wrong diagnosis from
receiving repeated credit. Reference observations attached directly to a
diagnosis still count as evidence but do not create an invented rationale step.

Every case remains in the evaluation, including wrong categories, abstentions,
unsupported KGs and runtime errors. Empty denominators score zero. A malformed
tree receives no valid predicted explanation steps or milestone credit; its
errors remain in the case report. Separate evidence metrics still inspect any
extracted observations. Micro scores pool counts; macro scores average case
scores. They must not be compared as if they were identical.

## Encoder and reproducibility

Encoder: [S-PubMedBert-MS-MARCO model card](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO),
revision 96786c7024f95c5aac7f2b9a18086c7b97b23036. The model card specifies
attention-masked mean pooling and a 350-token context. The evaluator uses
nonoverlapping token chunks for longer text, combines their token-weighted
representations, and normalizes the final vector; it does not silently discard
the end of a paragraph. This chunk aggregation is our implementation choice.
The encoder is about 0.41 GiB. Its repository declares CC-BY-NC-2.0; see its
model card for attribution and usage conditions.

Downloads verify the model weight SHA-256 against pinned repository metadata.
All file hashes and the repository revision are saved in download_manifest.json.
Every evaluation checks those hashes, loads only local files with remote code
disabled, and uses weights-only checkpoint loading. No sentence-transformers
or SciPy installation is required.

Outputs include summary.json (counts, macro/micro scores, configuration, model,
split/prediction/source hashes, time, and chunk statistics), report.md, and
per-case JSON containing the actual matched observation and rationale pairs.
Case files contain clinical text and belong under the same access conditions
as the original dataset.

## Interpretation

The reference audit in MODEL_BEHAVIOR_REPORT.md found two metadata-versus-root
disagreements in the broad 25-case cohort: COPD / Moderate COPD and Heart Failure
/ HFmrEF. Do not count a correct annotated subtype as a reasoning failure merely
because the folder-derived PDD is broader. New summaries expose both endpoints;
data/evaluation/d6_dev remains the historical output, and
data/evaluation/d6_dev_reference_audit contains the reconciled comparison.

These are practical annotation-agreement proxies, not an exact reproduction of
the paper's semantic judge. The threshold is explicitly uncalibrated. Similarity
can miss negation, measurements, temporal differences, or medically important
distinctions. Conversely, a sound alternative rationale may differ from the
annotation. Diagnosis labels therefore use explicit aliases rather than a loose
embedding match. No clinical correctness claim follows from a high embedding
score. The optional local LLM judge and human threshold calibration are deferred;
no extra research ablation or training is needed to use this harness.
