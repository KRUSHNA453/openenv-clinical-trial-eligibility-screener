"""Client for the Clinical Trial Eligibility Screener environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState


class ClinicalTrialEligibilityScreenerEnv(
    EnvClient[ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState]
):
    """Typed client wrapper for the screener environment."""

    def _step_payload(self, action: ClinicalTrialAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ClinicalTrialObservation]:
        observation = ClinicalTrialObservation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ClinicalTrialState:
        return ClinicalTrialState.model_validate(payload)
