"""Typed models for the Clinical Trial Eligibility Screener environment."""

from __future__ import annotations

from enum import Enum

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TrialDecision(str, Enum):
    PENDING = "PENDING"
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    NEED_MORE_INFO = "NEED_MORE_INFO"


class BloodPressure(BaseModel):
    systolic: int = Field(..., ge=50, le=260)
    diastolic: int = Field(..., ge=30, le=180)


class LabResults(BaseModel):
    creatinine_mg_dl: float = Field(..., ge=0.0)
    hba1c_percent: float = Field(..., ge=0.0, le=20.0)
    blood_pressure: BloodPressure


class PatientProfile(BaseModel):
    patient_id: str
    age: int = Field(..., ge=0, le=120)
    gender: str
    conditions: list[str] = Field(default_factory=list, description="Human-readable diagnoses")
    icd10_codes: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    lab_results: LabResults


class TrialProtocol(BaseModel):
    trial_id: str
    title: str
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)


class RewardBreakdown(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    criteria_match_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    correct_reasoning_tags: list[str] = Field(default_factory=list)
    missing_reasoning_tags: list[str] = Field(default_factory=list)
    hallucinated_claims: list[str] = Field(default_factory=list)
    decision_correct: bool | None = None
    step_penalty: float = Field(default=0.0, ge=0.0)
    notes: list[str] = Field(default_factory=list)


class ClinicalTrialAction(Action):
    thought: str = Field(
        ...,
        min_length=1,
        description="Free-form agent reasoning for the current screening step.",
    )
    decision: TrialDecision = Field(
        default=TrialDecision.PENDING,
        description="Current eligibility decision for the patient.",
    )


class ClinicalTrialObservation(Observation):
    case_id: str = Field(..., description="Identifier of the active screening case.")
    difficulty: DifficultyLevel
    patient_profile: PatientProfile
    trial_protocol: TrialProtocol
    medical_record_segment: str = Field(
        ...,
        description="Current text segment from the patient record.",
    )
    current_criterion: str = Field(
        ...,
        description="The criterion the agent should focus on for this step.",
    )
    remaining_criteria: list[str] = Field(default_factory=list)
    available_reasoning_tags: list[str] = Field(default_factory=list)
    decision_status: TrialDecision = Field(default=TrialDecision.PENDING)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)


class ClinicalTrialState(State):
    case_id: str | None = None
    difficulty: DifficultyLevel | None = None
    patient_profile: PatientProfile | None = None
    trial_protocol: TrialProtocol | None = None
    revealed_segment_ids: list[str] = Field(default_factory=list)
    review_cursor: int = Field(default=0, ge=0)
    max_steps: int = Field(default=12, ge=1)
    matched_reasoning_tags: list[str] = Field(default_factory=list)
    hallucinated_claims: list[str] = Field(default_factory=list)
    latest_decision: TrialDecision = Field(default=TrialDecision.PENDING)


class AgentPrediction(BaseModel):
    case_id: str
    final_decision: TrialDecision
    reasoning_tags: list[str] = Field(default_factory=list)
    steps_taken: int = Field(default=0, ge=0)
    final_reward: float | None = None
    transcript: list[dict[str, str | int | float | bool | None]] = Field(default_factory=list)


class GroundTruthRecord(BaseModel):
    case_id: str
    difficulty: DifficultyLevel
    expected_decision: TrialDecision
    eligible: bool
    reasoning_tags: list[str] = Field(default_factory=list)


class GradeReport(BaseModel):
    case_id: str
    expected_eligible: bool
    predicted_eligible: bool | None = None
    terminal_decision_submitted: bool = False
    decision_correct: bool = False
    expected_reasoning_tags: list[str] = Field(default_factory=list)
    matched_reasoning_tags: list[str] = Field(default_factory=list)
    missing_reasoning_tags: list[str] = Field(default_factory=list)
    unexpected_reasoning_tags: list[str] = Field(default_factory=list)
    reasoning_tag_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_tag_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_tag_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    decision_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_score: float = Field(default=0.0, ge=0.0, le=1.0)
