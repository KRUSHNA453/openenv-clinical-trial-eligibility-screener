"""Sample-format inference runner for the clinical trial screener."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import httpx
from openenv.core.client_types import StepResult
from openai import OpenAI
from dotenv import load_dotenv

from client import ClinicalTrialEligibilityScreenerEnv
from grader import grade_prediction
from models import AgentPrediction, ClinicalTrialAction, ClinicalTrialObservation, TrialDecision
from server.case_data import load_case_definitions


load_dotenv()


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("CLINICAL_TRIAL_TASK", "all")
BENCHMARK = os.getenv("CLINICAL_TRIAL_BENCHMARK", "clinical_trial_eligibility_screener")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "250"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are screening a patient for clinical trial eligibility.
    Respond with exactly one JSON object and no markdown.
    Required JSON keys:
    - thought: short grounded reasoning
    - decision: one of PENDING, ELIGIBLE, INELIGIBLE, NEED_MORE_INFO
    - reasoning_tags: array of allowed reasoning tags only
    Do not invent facts not present in the medical record segment, patient profile, or trial protocol.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN, API_KEY, or OPENAI_API_KEY")
    return API_KEY


def extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model response was empty")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    raise ValueError("Could not find a JSON object in model response")


def build_user_prompt(
    step: int,
    observation: ClinicalTrialObservation,
    last_reward: float,
    history: list[str],
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current criterion: {observation.current_criterion}
        Medical record segment: {observation.medical_record_segment}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Allowed reasoning tags: {", ".join(observation.available_reasoning_tags)}
        Patient profile JSON: {json.dumps(observation.patient_profile.model_dump(), ensure_ascii=True)}
        Trial protocol JSON: {json.dumps(observation.trial_protocol.model_dump(), ensure_ascii=True)}
        Return the next screening action as JSON.
        """
    ).strip()


def fallback_model_action() -> tuple[str, ClinicalTrialAction]:
    fallback_action = ClinicalTrialAction(
        thought="Need more info from the record.",
        decision=TrialDecision.NEED_MORE_INFO,
        metadata={"reasoning_tags": []},
    )
    return "decision=NEED_MORE_INFO|tags=none", fallback_action


def format_action_log(decision: TrialDecision, reasoning_tags: list[str]) -> str:
    tags = ",".join(reasoning_tags) if reasoning_tags else "none"
    return f"decision={decision.value}|tags={tags}"


def get_model_action(
    client: OpenAI,
    step: int,
    observation: ClinicalTrialObservation,
    last_reward: float,
    history: list[str],
) -> tuple[str, ClinicalTrialAction]:
    user_prompt = build_user_prompt(step, observation, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        payload = extract_json_payload(text)
        allowed_tags = set(observation.available_reasoning_tags)
        reasoning_tags = sorted(
            {
                tag.strip().lower()
                for tag in payload.get("reasoning_tags", [])
                if isinstance(tag, str) and tag.strip().lower() in allowed_tags
            }
        )
        decision = TrialDecision(payload.get("decision", TrialDecision.NEED_MORE_INFO.value))
        thought = str(payload.get("thought", "")).strip() or "Need more info from the record."
        action = ClinicalTrialAction(
            thought=thought,
            decision=decision,
            metadata={"reasoning_tags": reasoning_tags},
        )
        return format_action_log(decision, reasoning_tags), action
    except Exception as exc:  # pragma: no cover - exercised in live runs
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return fallback_model_action()


class HTTPEnvAdapter:
    """Fallback adapter that speaks to OpenEnv over HTTP instead of WebSockets."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=60.0)

    async def reset(self, **kwargs: Any) -> StepResult[ClinicalTrialObservation]:
        response = await self._client.post(f"{self._base_url}/reset", json=kwargs)
        response.raise_for_status()
        payload = response.json()
        return StepResult(
            observation=ClinicalTrialObservation.model_validate(payload["observation"]),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    async def step(self, action: ClinicalTrialAction) -> StepResult[ClinicalTrialObservation]:
        response = await self._client.post(
            f"{self._base_url}/step",
            json={"action": action.model_dump()},
        )
        response.raise_for_status()
        payload = response.json()
        return StepResult(
            observation=ClinicalTrialObservation.model_validate(payload["observation"]),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    async def close(self) -> None:
        await self._client.aclose()


async def connect_env() -> ClinicalTrialEligibilityScreenerEnv | HTTPEnvAdapter:
    if ENV_BASE_URL:
        env = ClinicalTrialEligibilityScreenerEnv(base_url=ENV_BASE_URL)
        try:
            await env.connect()
            return env
        except Exception as exc:  # pragma: no cover - depends on local websocket support
            print(
                f"[DEBUG] WebSocket connection failed, falling back to HTTP: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return HTTPEnvAdapter(ENV_BASE_URL)
    if LOCAL_IMAGE_NAME:
        return await ClinicalTrialEligibilityScreenerEnv.from_docker_image(LOCAL_IMAGE_NAME)
    raise RuntimeError("Set LOCAL_IMAGE_NAME/IMAGE_NAME or ENV_BASE_URL before running inference")


async def run_task(
    client: OpenAI,
    env: ClinicalTrialEligibilityScreenerEnv | HTTPEnvAdapter,
    case_id: str,
) -> dict[str, Any]:
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_decision = TrialDecision.NEED_MORE_INFO
    final_tags: list[str] = []
    transcript: list[dict[str, Any]] = []

    log_start(task=case_id, env=BENCHMARK, model=MODEL_NAME)

    result = await env.reset(case_id=case_id)
    try:
        observation = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_str, action = get_model_action(client, step, observation, last_reward, history)
            error = None

            try:
                result = await env.step(action)
            except Exception as exc:
                error = str(exc).replace("\n", " ")
                log_step(step=step, action=action_str, reward=0.0, done=True, error=error)
                break

            observation = result.observation
            reward = float(result.reward or 0.0)
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            final_decision = action.decision
            final_tags = list(action.metadata.get("reasoning_tags", []))

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            transcript.append(
                {
                    "step": step,
                    "action": action_str,
                    "reward": reward,
                    "done": done,
                    "criterion": observation.current_criterion,
                }
            )
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        prediction = AgentPrediction(
            case_id=case_id,
            final_decision=final_decision,
            reasoning_tags=final_tags,
            steps_taken=steps_taken,
            final_reward=rewards[-1] if rewards else 0.0,
            transcript=transcript,
        )
        grade = grade_prediction(prediction)
        score = min(max(float(grade.total_score), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        return {
            "task": case_id,
            "prediction": prediction.model_dump(),
            "grade": grade.model_dump(),
            "rewards": rewards,
            "score": score,
            "success": success,
        }
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def resolve_case_ids(task_name: str) -> list[str]:
    all_case_ids = [case.case_id for case in load_case_definitions()]
    if task_name == "all":
        return all_case_ids
    if task_name not in all_case_ids:
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {', '.join(all_case_ids)}")
    return [task_name]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default=TASK_NAME,
        help="Case id to run, or 'all' to run the full three-task benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "inference_summary.json",
        help="File path for a machine-readable summary",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=require_api_key())
    summaries = []
    env = await connect_env()
    try:
        for case_id in resolve_case_ids(args.task):
            summaries.append(await run_task(client, env, case_id))
    finally:
        try:
            await env.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"[DEBUG] env.close() error (container cleanup): {exc}", file=sys.stderr, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "api_base_url": API_BASE_URL,
                "task": args.task,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
