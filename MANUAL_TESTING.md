# Testing a clinical case locally

Use the project venv from this project directory. All inference is on-device.
The current model choice is recorded in data/models/selected_model.json when
assessment is complete. You can explicitly request base or d4 for inspection.

1. Copy examples/case_template.json and enter the fields:

| Field | Clinical content |
|---|---|
| input1 | Chief complaint |
| input2 | History of present illness |
| input3 | Past medical history |
| input4 | Family history |
| input5 | Physical examination |
| input6 | Pertinent laboratory, imaging and investigation results |

Empty fields are allowed; at least one field must contain clinical text.
An optional case_id labels the output. Do not add an expected diagnosis or
category field. Historical diagnoses within the actual note are valid inputs.

2. Start the local model server. To make both base and D4 available:

```powershell
.\scripts\start_local_model.ps1 -ModelSize 4B -AdapterPath data/models/d4_best.gguf
```

If a server is already running with a different model configuration, stop it
with scripts/stop_local_model.ps1 before starting the requested one. Wait until
active training/evaluation finishes. Manual testing refuses to change adapter
settings during a budgeted GPU job so it cannot interfere with comparisons.

3. Run a case:

```powershell
.\venv\Scripts\python.exe -B scripts\test_case.py examples/your_case.json
```

The command uses the recorded selected model; use --model base or --model d4
to explicitly choose. D4 requires its adapter to be loaded in the server.

For a repeatable example drawn from the development split:

```powershell
.\venv\Scripts\python.exe -B scripts\test_case.py examples/direct_dev_case.json
```

This example is not an independent test of model quality. Its expected answer
is not included in the input JSON.

Outputs are saved in data/manual_runs/<timestamp>/:

- case.json: supplied clinical fields.
- result.json: category, proposed diagnosis, observations, reasoning nodes,
  warnings/errors, runtime and adapter configuration.
- report.html: standalone readable report with source passages, rationales,
  diagnostic milestones and expandable original clinical text. It has no
  external assets or network calls.

Open report.html in a browser. The report is a view of the model output, not an
independent verification of its medical reasoning. The model can misclassify
the category, choose the wrong subtype, or stop with insufficient evidence.
Gastritis has no supplied KG; the current KG pipeline covers only 24 categories.
An unsupported/unfamiliar disease can still be misrouted to a known category.

Stop the model server after use to release GPU memory:

```powershell
.\scripts\stop_local_model.ps1
```

This interface runs the existing multi-stage pipeline. A browser input form
and a separately trained single-pass full-tree demo remain future work.
