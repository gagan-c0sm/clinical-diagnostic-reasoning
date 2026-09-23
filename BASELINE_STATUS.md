# Local baseline checkpoint — October 28, 2026

Patient-grouped splits and an operational local pipeline are implemented.
All six integrity/failure-handling unit tests pass. Ten-note live outputs for
both models passed source-passage and tree-integrity checks.

## Practical model choice

| Model (Q4_K_M) | Category matches | Diagnosis matches | Mean seconds/note |
|---|---:|---:|---:|
| Qwen3-4B | 8/10 | 8/10 | 10.9 |
| Qwen3-8B | 8/10 | 5/10 | 14.3 |

Keep **4B as the provisional default**: it gave better diagnosis matching and
lower latency on this development check. Both use the same corrected prompt,
source-passage selection, non-thinking mode, and 8,192-token context.
Observed total GPU usage was about 4 GiB (4B) and 6 GiB (8B).

These ten cases are one annotation each from the first ten alphabetically
ordered categories in the development split, not a representative accuracy
estimate. Prompts were tuned on these cases. The 80-annotation final test set
was not used. Do not interpret 8/10 as established general accuracy or conclude
that 4B is universally superior to 8B.

## What improved

The initial prompt incorrectly restricted category selection to the acute
admission reason. Allowing evaluated chronic disease and prioritizing specific
findings increased 4B diagnosis matches from 3/10 to 8/10 on the same cases.
Free-form quote generation was replaced with selection of numbered original
passages. This guarantees retained text comes from the note and substantially
reduces generated tokens. It does not guarantee correct evidence selection or
medical reasoning.

## Next useful work

Expand the development check across all 25 categories, inspect wrong routing
and unsupported reasoning, and improve category selection before deciding on
fine-tuning. The current 4B failures on this small batch route ACS and COPD to
Heart Failure. Do not add case-specific rules to force these labels.

D3/D4 scripts, training, adapters and synthetic augmentation remain unimplemented.
A visual demo and semantic explanation scoring are also pending. No cloud LLM
inference or global package installation was used.

## Known dataset limitations

Gastritis lacks a supplied KG (27 annotations). Four Severe Asthma Exacerbation
labels are not silently equated with Severe Asthma. Two source observations
have no rationale node. The original split is historical; use data/splits_v2.

Detailed predictions and source/version hashes are stored in
`data/runs/dev_4b_corrected_routing_10/` and
`data/runs/dev_8b_corrected_routing_10/`.
