# Agent Run Protocol

This file defines the required bookkeeping after every serious experiment, Colab run, checkpoint evaluation, or Kaggle score.

## Trigger

Use this protocol when the user mentions:

- a finished run
- a Kaggle public score
- a run bundle
- a submitted `submission.zip`
- generated-eval results
- checkpoint results
- a new experiment archive
- a failed or interrupted run

## Step 1: Identify The Run

Collect or infer:

- `EXPERIMENT_NAME`
- notebook used
- model
- LoRA rank and alpha
- LoRA target modules
- training rows
- eval rows
- target format
- max sequence length
- max new tokens
- learning rate
- epochs
- run bundle path
- submission zip path
- public score, if available
- local generated-eval score, if available
- family-level scores, if available

If a value is missing, write `unknown`, not a guessed value.

## Step 2: Archive Required Artifacts

For a scored Kaggle submission, ensure there is one folder under:

```text
data/outputs/submissions/
```

The folder name should contain:

```text
YYYY-MM-DD_method_score_X_XX
```

It should contain, when available:

- submitted `submission.zip`
- source `{EXPERIMENT_NAME}_run_bundle.zip`
- `metadata.json`
- `adapter_config.json`
- `zip_contents.txt`
- generated-eval files
- checkpoint-eval files
- probe-evolution files

Do not mix diagnostics into the Kaggle upload zip.

## Step 3: Update Tracking Files

After every serious run, update:

1. `doc/EXPERIMENT_CHECKLIST.md`
2. `doc/SUBMISSION_TRACKING.md` if a Kaggle submission was scored
3. `doc/LOCAL_PROJECT_MEMORY.md`
4. `doc/PROJECT_DECISION_LOG.md` if the result changes a durable decision
5. `doc/EXPERIMENT_REGISTRY.csv` for all serious runs if present
6. `data/outputs/submissions/submissions_registry.csv` for scored Kaggle submissions if present

If a method-ladder file exists, do not rewrite it as a checklist. Add only a short note when a ladder idea was tested, rejected, or promoted into the decision log.

## Step 4: Decide The Run Status

Classify the run as one of:

- `win`: beats best public score or clearly improves target weak families
- `useful_negative`: failed but gives strong evidence
- `inconclusive`: missing artifacts or noisy signal
- `failed_runtime`: runtime, packaging, OOM, disconnect, or notebook failure
- `superseded`: replaced by a better run or better method

## Step 5: Update The Next Action

End the update with one explicit next action:

- continue same method with one small change
- select best checkpoint
- archive and stop this method
- escalate to procedural supervision
- escalate to offline sampling
- escalate to radical pivot
- run diagnostic only before training

Do not leave the next action vague.
