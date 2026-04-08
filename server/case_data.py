"""Shared case and ground-truth loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from models import DifficultyLevel, GroundTruthRecord, PatientProfile, TrialDecision, TrialProtocol


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    difficulty: DifficultyLevel
    patient_profile: PatientProfile
    trial_protocol: TrialProtocol
    record_segments: tuple[str, ...]
    criterion_sequence: tuple[str, ...]
    expected_decision: TrialDecision
    supporting_reasoning_tags: frozenset[str]
    supported_facts: frozenset[str]
    summary: str


DATA_DIR = Path(__file__).resolve().parent / "data"
CASES_PATH = DATA_DIR / "cases.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"


@lru_cache(maxsize=1)
def load_case_definitions() -> tuple[CaseDefinition, ...]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases: list[CaseDefinition] = []
    for item in payload:
        cases.append(
            CaseDefinition(
                case_id=item["case_id"],
                difficulty=DifficultyLevel(item["difficulty"]),
                patient_profile=PatientProfile.model_validate(item["patient_profile"]),
                trial_protocol=TrialProtocol.model_validate(item["trial_protocol"]),
                record_segments=tuple(item["record_segments"]),
                criterion_sequence=tuple(item["criterion_sequence"]),
                expected_decision=TrialDecision(item["expected_decision"]),
                supporting_reasoning_tags=frozenset(item["supporting_reasoning_tags"]),
                supported_facts=frozenset(item["supported_facts"]),
                summary=item["summary"],
            )
        )
    return tuple(cases)


@lru_cache(maxsize=1)
def load_case_index() -> dict[str, CaseDefinition]:
    return {case.case_id: case for case in load_case_definitions()}


@lru_cache(maxsize=1)
def load_ground_truth_index() -> dict[str, GroundTruthRecord]:
    payload = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return {
        item["case_id"]: GroundTruthRecord.model_validate(item)
        for item in payload
    }
