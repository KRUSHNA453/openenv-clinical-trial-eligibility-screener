# Evaluator Guide

This repository is set up so an evaluator can verify the important pieces quickly.

## Fastest Review Path

Run these commands from the repo root:

```powershell
uv sync --extra dev
uv run python scripts/quick_check.py
```

That script runs:

- `openenv validate`
- `pytest`
- `ruff check`
- the deterministic rule baseline
- a data integrity check that confirms the benchmark has easy, medium, and hard tasks

## Files That Matter Most

- `README.md`: project overview, setup, action space, observation space, baseline scores
- `openenv.yaml`: OpenEnv app metadata
- `server/env.py`: environment logic, rewards, episode flow
- `server/app.py`: FastAPI/OpenEnv server
- `server/data/cases.json`: benchmark tasks
- `server/data/ground_truth.json`: deterministic grading targets
- `models.py`: typed Pydantic models
- `grader.py`: deterministic grader
- `baseline.py`: reproducible baseline
- `inference.py`: OpenAI-client inference runner with strict `[START]`, `[STEP]`, `[END]` logs
- `Dockerfile`: container entrypoint for deployment
- `scripts/validate-submission.sh`: organizer-style pre-submission validator

## What Needs Secrets

These checks do not need secrets:

- `uv run python scripts/quick_check.py`
- `uv run openenv validate .`
- `uv run pytest`
- `uv run python baseline.py --json`

These flows do need secrets or external services:

- `inference.py` with a live model endpoint
- Hugging Face Spaces deployment
- organizer ping validation against a live Space URL

## Local Review Commands

Validate the OpenEnv package:

```powershell
uv run openenv validate .
```

Run the baseline:

```powershell
uv run python baseline.py --json
```

Run inference against a local server:

```powershell
uv run server
$env:ENV_BASE_URL="http://127.0.0.1:8000"
uv run python inference.py --task all
```

Run the organizer-style validator after deployment:

```bash
bash scripts/validate-submission.sh https://your-space.hf.space
```
