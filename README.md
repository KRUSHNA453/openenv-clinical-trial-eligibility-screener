---
title: Clinical Trial Eligibility Screener
emoji: stethoscope
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - clinical-trials
  - healthcare
---

# Clinical Trial Eligibility Screener

OpenEnv environment for training and evaluating agents that screen patient medical
records against clinical trial inclusion and exclusion criteria.

## Quick Start For Evaluators

If you want the shortest possible review path, run:

```powershell
uv sync --extra dev
uv run python scripts/quick_check.py
```

That covers the local checks most relevant to evaluation. A shorter human-readable review guide
is also available in [EVALUATOR_GUIDE.md](./EVALUATOR_GUIDE.md).

## What Is Included

- OpenEnv-compatible `reset()`, `step()`, and `state()` environment
- Typed Pydantic models for action, observation, state, reward breakdown, predictions, and grading
- Three deterministic tasks covering easy, medium, and hard screening scenarios
- Partial-credit reward logic with hallucination penalties and step penalties after 10 steps
- Deterministic grader that compares final eligibility and reasoning tags against ground truth JSON
- OpenAI-compatible `inference.py` that reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- Rule-based `baseline.py` and pytest coverage
- Dockerfiles for OpenEnv and Hugging Face Spaces style deployment

## Scenario Design

Each episode gives the agent:

- A structured patient profile with age, gender, conditions, ICD-10 codes, medications, and labs
- A clinical trial protocol with inclusion and exclusion criteria
- A rolling medical record segment and the current criterion under review

Current benchmark cases:

- `easy_t2d_adult`: adult with Type 2 diabetes and no listed exclusions
- `medium_ckd_exclusion`: patient meets glycemic inclusion but is excluded by kidney disease and creatinine
- `hard_drug_interaction`: mixed profile with qualifying labs but disqualifying drug-drug interactions

## Action Space

`ClinicalTrialAction`

- `thought`: grounded free-text reasoning for the current step
- `decision`: `PENDING`, `ELIGIBLE`, `INELIGIBLE`, or `NEED_MORE_INFO`
- `metadata.reasoning_tags`: optional structured tags for deterministic grading

## Observation Space

`ClinicalTrialObservation`

- `case_id`
- `difficulty`
- `patient_profile`
- `trial_protocol`
- `medical_record_segment`
- `current_criterion`
- `remaining_criteria`
- `available_reasoning_tags`
- `decision_status`
- `reward_breakdown`
- `reward`
- `done`

## Reward Logic

Environment reward is `0.0` to `1.0`.

- `75%` of the reward comes from recall over expected reasoning tags
- `25%` comes from the final terminal decision being correct
- hallucinated unsupported facts reduce reward
- each step after step `10` adds a penalty

The deterministic grader in [grader.py](./grader.py) is stricter than the environment reward:

- it requires a terminal final decision
- it scores decision correctness
- it scores reasoning tags with precision/recall F1, so extra tags are penalized

## Ground Truth

Shared benchmark data lives in:

- [server/data/cases.json](./server/data/cases.json)
- [server/data/ground_truth.json](./server/data/ground_truth.json)

The environment loads `cases.json`. The grader uses `ground_truth.json`.

## Inference Runner

[inference.py](./inference.py) uses the OpenAI Python client against any OpenAI-compatible endpoint.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`
- `LOCAL_IMAGE_NAME` or `IMAGE_NAME` when running against a local Docker image

Optional environment variable:

- `ENV_BASE_URL` to connect to an already-running environment instead of starting Docker

Example:

```powershell
$env:API_BASE_URL="https://your-openai-compatible-endpoint/v1"
$env:MODEL_NAME="your-model-name"
$env:HF_TOKEN="your-token"
$env:LOCAL_IMAGE_NAME="clinical-trial-screener:latest"
uv run python inference.py --task hard_drug_interaction
```

You can also copy [.env.example](./.env.example) to `.env` and fill in your values. `inference.py`
loads `.env` automatically if it is present, and `.env` is git-ignored so secrets are not committed.

Stdout format is intentionally strict and only emits single-line:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

By default, the script runs all three benchmark tasks and writes a machine-readable
summary to `artifacts/inference_summary.json`.

## Deterministic Grading

Grade a saved prediction file or an inference log file:

```powershell
uv run python grader.py artifacts/last_prediction.json
uv run python grader.py logs\episode.log
```

## Baseline Scores

Verified locally with:

```powershell
uv run python baseline.py --json
```

Measured baseline results on the current three-case benchmark:

- `easy_t2d_adult`: reward `1.000`, grader score `0.944`
- `medium_ckd_exclusion`: reward `1.000`, grader score `0.917`
- `hard_drug_interaction`: reward `1.000`, grader score `0.929`
- average environment reward: `1.000`
- average deterministic grader score: `0.930`

The reward is perfect because the rule baseline hits all required signals for the environment reward function. The grader score is lower because the baseline emits a few extra reasoning tags, and the grader penalizes that with F1 scoring.

## Validation And Tests

```powershell
uv sync --extra dev
uv run openenv validate .
uv run pytest
uv run ruff check .
```

Organizer-style pre-submission validator script:

```bash
bash scripts/validate-submission.sh https://your-space.hf.space
```

Fast evaluator-oriented local check:

```powershell
uv run python scripts/quick_check.py
```

## Running The Server

```powershell
uv run server
```

Or with Uvicorn:

```powershell
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

The root [Dockerfile](./Dockerfile) is the canonical container definition used for local
builds and deployment.

Build locally:

```powershell
docker build -t clinical-trial-screener .
```

## Project Structure

```text
.
|-- Dockerfile
|-- EVALUATOR_GUIDE.md
|-- README.md
|-- baseline.py
|-- client.py
|-- grader.py
|-- inference.py
|-- models.py
|-- openenv.yaml
|-- pyproject.toml
|-- scripts/
|   |-- quick_check.py
|   `-- validate-submission.sh
|-- tests/
`-- server/
    |-- app.py
    |-- case_data.py
    |-- data/
    |   |-- cases.json
    |   `-- ground_truth.json
    `-- env.py
```
