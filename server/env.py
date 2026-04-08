"""Core OpenEnv environment for clinical trial eligibility screening."""

from __future__ import annotations

import random
import re
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from server.case_data import CaseDefinition, load_case_definitions
from models import (
    ClinicalTrialAction,
    ClinicalTrialObservation,
    ClinicalTrialState,
    DifficultyLevel,
    RewardBreakdown,
    TrialDecision,
)


TAG_PATTERN = re.compile(r"#([a-z0-9_]+)")

TRACKED_FACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "warfarin": re.compile(r"\b(?:taking|on)\s+warfarin\b", re.IGNORECASE),
    "clarithromycin": re.compile(r"\b(?:taking|on)\s+clarithromycin\b", re.IGNORECASE),
    "kidney_disease": re.compile(
        r"\b(?:has|history of|with)\s+(?:chronic\s+)?kidney disease\b",
        re.IGNORECASE,
    ),
    "type_1_diabetes": re.compile(r"\b(?:has|with)\s+type\s+1 diabetes\b", re.IGNORECASE),
    "pregnant": re.compile(r"\b(?:is|patient is)\s+pregnant\b", re.IGNORECASE),
}

class ClinicalTrialEligibilityScreenerEnvironment(
    Environment[ClinicalTrialAction, ClinicalTrialObservation, ClinicalTrialState]
):
    """Deterministic clinical trial eligibility screening environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 12
    SOFT_STEP_LIMIT = 10
    AVAILABLE_REASONING_TAGS = [
        "adult_patient",
        "age_in_range",
        "has_type_2_diabetes",
        "no_type_1_diabetes",
        "not_pregnant",
        "hba1c_under_7",
        "hba1c_in_range",
        "creatinine_high",
        "creatinine_within_limit",
        "kidney_disease_history",
        "no_stage3_kidney_disease",
        "warfarin_interaction",
        "cyp3a4_inhibitor_interaction",
        "systolic_bp_within_limit",
        "no_recent_pancreatitis",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._cases = list(load_case_definitions())
        self._cases_by_difficulty: dict[DifficultyLevel, list[CaseDefinition]] = {
            difficulty: [case for case in self._cases if case.difficulty == difficulty]
            for difficulty in DifficultyLevel
        }
        self._reset_counter = 0
        self._active_case: CaseDefinition | None = None
        self._submitted_tags: set[str] = set()
        self._hallucinated_claims: list[str] = []
        self._last_reward = RewardBreakdown()
        self._state = ClinicalTrialState(episode_id=str(uuid4()), max_steps=self.MAX_STEPS)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        difficulty: str | None = None,
        case_id: str | None = None,
        **_: Any,
    ) -> ClinicalTrialObservation:
        """Reset the environment to a new patient/trial screening episode."""
        self._reset_rubric()

        case = self._select_case(seed=seed, difficulty=difficulty, case_id=case_id)
        current_episode_id = episode_id or str(uuid4())
        self._active_case = case
        self._submitted_tags = set()
        self._hallucinated_claims = []
        self._last_reward = RewardBreakdown(
            notes=["Episode reset. Submit reasoning tags in metadata or as #tags in thought."]
        )
        self._state = ClinicalTrialState(
            episode_id=current_episode_id,
            step_count=0,
            case_id=case.case_id,
            difficulty=case.difficulty,
            patient_profile=case.patient_profile,
            trial_protocol=case.trial_protocol,
            revealed_segment_ids=["segment-1"],
            review_cursor=0,
            max_steps=self.MAX_STEPS,
            latest_decision=TrialDecision.PENDING,
        )
        return self._build_observation(
            decision_status=TrialDecision.PENDING,
            done=False,
            reward=0.0,
            reward_breakdown=self._last_reward,
        )

    def step(
        self,
        action: ClinicalTrialAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> ClinicalTrialObservation:
        """Advance the review session with the agent's reasoning and decision."""
        del timeout_s

        case = self._require_active_case()
        if self._state.step_count >= self.MAX_STEPS:
            return self._build_observation(
                decision_status=self._state.latest_decision,
                done=True,
                reward=self._last_reward.score,
                reward_breakdown=self._last_reward,
            )

        self._state.step_count += 1
        self._state.latest_decision = action.decision

        extracted_tags = self._extract_reasoning_tags(action)
        self._submitted_tags.update(extracted_tags)

        new_hallucinations = self._detect_hallucinated_claims(action.thought, case)
        for claim in new_hallucinations:
            if claim not in self._hallucinated_claims:
                self._hallucinated_claims.append(claim)

        if action.decision in {TrialDecision.PENDING, TrialDecision.NEED_MORE_INFO}:
            self._advance_cursor()

        self._state.matched_reasoning_tags = sorted(
            self._submitted_tags.intersection(case.supporting_reasoning_tags)
        )
        self._state.hallucinated_claims = list(self._hallucinated_claims)

        is_terminal_decision = action.decision in {
            TrialDecision.ELIGIBLE,
            TrialDecision.INELIGIBLE,
        }
        done = is_terminal_decision or self._state.step_count >= self.MAX_STEPS

        reward_breakdown = self._score_submission(
            submitted_tags=self._submitted_tags,
            decision=action.decision,
            step_count=self._state.step_count,
            hallucinated_claims=self._hallucinated_claims,
            expected_decision=case.expected_decision,
            expected_tags=case.supporting_reasoning_tags,
        )
        reward = reward_breakdown.score
        self._last_reward = reward_breakdown

        return self._build_observation(
            decision_status=action.decision,
            done=done,
            reward=reward,
            reward_breakdown=reward_breakdown,
        )

    @property
    def state(self) -> ClinicalTrialState:
        """Return the current episode state."""
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Clinical Trial Eligibility Screener",
            description=(
                "Evaluates whether an agent can screen patient records against "
                "structured clinical trial inclusion and exclusion criteria."
            ),
            version="0.1.0",
        )

    def _build_observation(
        self,
        *,
        decision_status: TrialDecision,
        done: bool,
        reward: float,
        reward_breakdown: RewardBreakdown,
    ) -> ClinicalTrialObservation:
        case = self._require_active_case()
        segment_index = min(self._state.review_cursor, len(case.record_segments) - 1)
        criterion_index = min(self._state.review_cursor, len(case.criterion_sequence) - 1)

        return ClinicalTrialObservation(
            case_id=case.case_id,
            difficulty=case.difficulty,
            patient_profile=case.patient_profile,
            trial_protocol=case.trial_protocol,
            medical_record_segment=case.record_segments[segment_index],
            current_criterion=case.criterion_sequence[criterion_index],
            remaining_criteria=list(case.criterion_sequence[criterion_index + 1 :]),
            available_reasoning_tags=self.AVAILABLE_REASONING_TAGS,
            decision_status=decision_status,
            reward_breakdown=reward_breakdown,
            done=done,
            reward=reward,
            metadata={
                "summary": case.summary,
                "revealed_segments": list(self._state.revealed_segment_ids),
                "step_count": self._state.step_count,
                "soft_step_limit": self.SOFT_STEP_LIMIT,
                "max_steps": self.MAX_STEPS,
            },
        )

    def _select_case(
        self,
        *,
        seed: int | None,
        difficulty: str | None,
        case_id: str | None,
    ) -> CaseDefinition:
        if case_id is not None:
            for case in self._cases:
                if case.case_id == case_id:
                    return case
            raise ValueError(f"Unknown case_id '{case_id}'")

        if difficulty is not None:
            level = DifficultyLevel(difficulty.lower())
        else:
            level = list(DifficultyLevel)[self._reset_counter % len(DifficultyLevel)]
            self._reset_counter += 1

        case_pool = self._cases_by_difficulty[level]
        if seed is None:
            return case_pool[0]

        rng = random.Random(seed)
        return case_pool[rng.randrange(len(case_pool))]

    def _advance_cursor(self) -> None:
        case = self._require_active_case()
        next_cursor = min(self._state.review_cursor + 1, len(case.record_segments) - 1)
        self._state.review_cursor = next_cursor
        segment_id = f"segment-{next_cursor + 1}"
        if segment_id not in self._state.revealed_segment_ids:
            self._state.revealed_segment_ids.append(segment_id)

    def _extract_reasoning_tags(self, action: ClinicalTrialAction) -> set[str]:
        tags: set[str] = set()
        metadata_tags = action.metadata.get("reasoning_tags", [])
        if isinstance(metadata_tags, list):
            for tag in metadata_tags:
                if isinstance(tag, str):
                    tags.add(tag.strip().lower())
        for match in TAG_PATTERN.findall(action.thought.lower()):
            tags.add(match.strip())
        return {tag for tag in tags if tag}

    def _detect_hallucinated_claims(
        self,
        thought: str,
        case: CaseDefinition,
    ) -> list[str]:
        claims: list[str] = []
        normalized = thought.strip()
        if not normalized:
            return claims

        for fact_key, pattern in TRACKED_FACT_PATTERNS.items():
            if pattern.search(normalized) and fact_key not in case.supported_facts:
                claims.append(f"Claimed unsupported fact: {fact_key}")
        return claims

    def _score_submission(
        self,
        *,
        submitted_tags: set[str],
        decision: TrialDecision,
        step_count: int,
        hallucinated_claims: list[str],
        expected_decision: TrialDecision,
        expected_tags: frozenset[str],
    ) -> RewardBreakdown:
        matched_tags = sorted(submitted_tags.intersection(expected_tags))
        missing_tags = sorted(expected_tags.difference(submitted_tags))
        criteria_ratio = len(matched_tags) / len(expected_tags)

        decision_correct: bool | None
        decision_bonus = 0.0
        if decision in {TrialDecision.ELIGIBLE, TrialDecision.INELIGIBLE}:
            decision_correct = decision == expected_decision
            decision_bonus = 0.25 if decision_correct else 0.0
        else:
            decision_correct = None

        hallucination_penalty = min(0.25, 0.05 * len(hallucinated_claims))
        step_penalty = max(0.0, (step_count - self.SOFT_STEP_LIMIT) * 0.03)

        raw_score = (0.75 * criteria_ratio) + decision_bonus - hallucination_penalty - step_penalty
        score = max(0.0, min(1.0, raw_score))

        notes = [
            f"Matched {len(matched_tags)} of {len(expected_tags)} expected reasoning tags.",
        ]
        if hallucinated_claims:
            notes.append("Unsupported claims were penalized.")
        if step_penalty > 0:
            notes.append("A step penalty was applied for exceeding 10 steps.")
        if decision_correct is False:
            notes.append("Final eligibility decision was incorrect.")

        return RewardBreakdown(
            score=score,
            criteria_match_ratio=criteria_ratio,
            correct_reasoning_tags=matched_tags,
            missing_reasoning_tags=missing_tags,
            hallucinated_claims=list(hallucinated_claims),
            decision_correct=decision_correct,
            step_penalty=step_penalty,
            notes=notes,
        )

    def _require_active_case(self) -> CaseDefinition:
        if self._active_case is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")
        return self._active_case
