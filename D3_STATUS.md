# D3 preparation and GPU trial — October 28, 2026

## Resumable session update

`scripts/train_d3_session.py` now implements deterministic shuffled data order,
completion-only loss, gradient accumulation, warmup/linear learning-rate decay,
periodic source-balanced validation, early stopping, a wall-time cap, and atomic
checkpoint directories containing adapter, optimizer, scheduler, random-number
states and data cursor. Resume rejects changed model/data/configuration and
validation cohorts. `status.json` reports current progress and the last durable
checkpoint separately. Partial microbatches are not saved as completed updates.

Resume was tested by stopping after two updates and resuming to four, then
comparing against an uninterrupted four-update run. All 504 adapter tensors,
history, scheduler and data cursor matched exactly. See
`data/training/resume_verification.json`. All 12 tests and `pip check` pass.

A bounded pilot was launched at `data/training/d3_pilot_200`: 200 update limit,
batch size 1, gradient accumulation 4, 2,048 token limit, validation on 32 fixed
source-balanced examples every 50 updates, checkpoint every validation, one-hour
wall-time cap, and early stopping after three non-improving checks. At most 800
training examples are consumed; this is an initial pilot rather than a full
pass through 31,754 examples. Its completion must be checked in `status.json`.
Logs are in `data/training/logs/d3_pilot_200.stdout.log` and `.stderr.log`.
The process remains local and does not register a Windows startup task.

To resume a bounded stop, use the original run configuration and the checkpoint
named in `latest.json`, for example:

```powershell
.\venv\Scripts\python.exe -u -B scripts\train_d3_session.py --output data/training/d3_pilot_200 --resume data/training/d3_pilot_200/checkpoint-000050 --steps 200 --grad-accum 4 --val-every 50 --val-samples 32 --max-hours 1
```

Use the actual latest checkpoint number. Do not start two trainers on this GPU.
Select the best validation checkpoint recorded in status, which may differ from
the latest checkpoint. Pilot validation loss is not a diagnostic accuracy score;
evaluate the selected adapter before deploying it or proceeding to D4.

The historical preparation/trial checkpoint follows below.

The supplementary mixture, compatible local training stack, trainable checkpoint,
and short QLoRA save/reload trials are verified. Full Phase-1 training has not run.
The trial adapters are debugging artifacts, not deployable medical specialists.

## Prepared data

`data/d3` contains 31,754 train, 1,489 validation, and 897 supplementary test
examples. Medical-O1 validation is assigned deterministically by normalized
question hash (approximately 5%). MedCaseReasoning retains its native splits.
Held-out questions take precedence over training copies. Preparation removes
27 duplicate rows and excludes 21 conflicting training questions. These checks
cover exact normalized inputs, not semantic paraphrases. Source metadata is
preserved. DiReCT evaluation cases are not included in this mixture.

Training lengths: median 623 tokens, p95 907, maximum 1,563. All training examples
fit a 2,048-token limit without truncation; 31,295/31,754 fit 1,024. Validation
maximum is 1,955. Tokenized examples retain complete inputs and outputs; prompt
tokens are masked from loss. The Qwen non-thinking chat prefix is used, with
the dataset's narrative reasoning and answer as the supervised completion.

See `data/d3/manifest.json` and `token_profile.json` for detailed counts.

## Environment and checkpoint

All libraries are inside the project venv: PyTorch 2.8.0+cu128, Transformers
4.56.2, PEFT 0.17.1, bitsandbytes 0.48.1, Accelerate 1.10.1, datasets 4.1.1.
The dataset library was aligned with Transformers' Hub dependency. `pip check`
passes. `data/d3/environment-lock.txt` captures the installed packages.
CUDA NF4 backward propagation and AdamW8bit were exercised on the actual GPU.

The official Qwen/Qwen3-4B checkpoint is in `data/models/Qwen3-4B-HF`, pinned to
revision 1cfa9a7208912126459214e8b04321603b3df60c. Weight hashes are verified.
The original inference GGUF files remain available as the untrained baseline.

## Verified trials

NF4 + double quantization, bf16 compute, rank 8 / alpha 16 LoRA on all seven
attention/MLP projections, gradient checkpointing, batch size 1, AdamW8bit.
Only the 16,515,072 LoRA parameters are trainable.

| Trial limit | Actual example lengths | Optimizer steps | Peak allocated | Peak reserved | Training seconds |
|---|---|---:|---:|---:|---:|
| 1,024 | 1,024 / 1,024 | 2 | 4,802 MiB | 5,826 MiB | 6.1 |
| 2,048 | 1,563 / 1,551 | 2 | 5,907 MiB | 7,356 MiB | 11.0 |

Both trials saved adapters, reloaded them with matching base preparation,
verified exact adapter tensor equality, and reproduced the same reference loss.
The initial failed reload check is retained under `d3_trial_1024`; it was fixed
by using identical k-bit base preparation during reload. Verified results are
under `data/training/d3_trial_1024_verified` and `d3_trial_2048_verified`.

The longer trial covers the longest current training examples; it does not
demonstrate backward propagation on a full 2,048-token artificial sequence.
Allocator reservation is close to the 8 GB limit; use batch size 1 and stop the
inference server before training. Training processes have exited, freeing VRAM.
Losses from two steps establish a functional training path, not accuracy gains.

## Run or repeat

```powershell
.\venv\Scripts\python.exe -u -B scripts\train_d3.py --steps 2 --seq-len 2048 --output data/training/new_trial
```

Existing output directories are protected from overwriting. The default is a
two-step trial; these trials choose long examples to stress memory. Longer runs
shuffle examples with seed 42. Full-session scheduling, periodic validation,
resumable optimizer checkpoints, and training duration selection should be
completed before an hours-long D3 run. The current script saves the adapter and
trial report at the end; it is not yet a resumable full-training controller.

All 12 project tests pass, including split isolation and completion-loss masking.
