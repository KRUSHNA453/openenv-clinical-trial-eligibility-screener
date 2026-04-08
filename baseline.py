"""Simple rule-based baseline for the clinical trial screener."""

from __future__ import annotations

import argparse
import json
from statistics import mean

from grader import grade_prediction
from models import AgentPrediction, ClinicalTrialAction, ClinicalTrialObservation, TrialDecision
from server.case_data import load_case_definitions
from server.env import ClinicalTrialEligibilityScreenerEnvironment


def _contains_any(values: list[str], needle: str) -> bool:
    return any(needle in value.lower() for value in values)


def derive_reasoning_tags(observation: ClinicalTrialObservation) -> list[str]:
    patient = observation.patient_profile
    protocol_text = " ".join(
        observation.trial_protocol.inclusion_criteria + observation.trial_protocol.exclusion_criteria
    ).lower()
    tags: set[str] = set()

    if patient.age > 18 and "greater than 18" in protocol_text:
        tags.add("adult_patient")
    if ("18 to 75" in protocol_text and 18 <= patient.age <= 75) or (
        "40 to 70" in protocol_text and 40 <= patient.age <= 70
    ):
        tags.add("age_in_range")
    if _contains_any(patient.conditions, "type 2 diabetes"):
        tags.add("has_type_2_diabetes")
    if not _contains_any(patient.conditions, "type 1 diabetes"):
        tags.add("no_type_1_diabetes")
    if not _contains_any(patient.conditions, "pregnan"):
        tags.add("not_pregnant")

    hba1c = patient.lab_results.hba1c_percent
    creatinine = patient.lab_results.creatinine_mg_dl
    systolic_bp = patient.lab_results.blood_pressure.systolic

    if "less than 7.0%" in protocol_text and hba1c < 7.0:
        tags.add("hba1c_under_7")
    if "between 7.0% and 9.0%" in protocol_text and 7.0 <= hba1c <= 9.0:
        tags.add("hba1c_in_range")
    if "greater than 1.5" in protocol_text and creatinine > 1.5:
        tags.add("creatinine_high")
    if "less than 1.5" in protocol_text and creatinine < 1.5:
        tags.add("creatinine_within_limit")
    if "less than 150" in protocol_text and systolic_bp < 150:
        tags.add("systolic_bp_within_limit")

    lower_conditions = [condition.lower() for condition in patient.conditions]
    lower_medications = [medication.lower() for medication in patient.medications]
    if any("kidney disease" in condition for condition in lower_conditions):
        tags.add("kidney_disease_history")
    if not any(
        "kidney disease" in condition and any(stage in condition for stage in ("stage 3", "stage 4", "stage 5"))
        for condition in lower_conditions
    ):
        tags.add("no_stage3_kidney_disease")
    if "warfarin" in lower_medications:
        tags.add("warfarin_interaction")
    if "clarithromycin" in lower_medications and "cyp3a4" in protocol_text:
        tags.add("cyp3a4_inhibitor_interaction")
    if any("remote pancreatitis" in condition for condition in lower_conditions):
        tags.add("no_recent_pancreatitis")

    return sorted(tags)


def derive_decision(reasoning_tags: list[str], observation: ClinicalTrialObservation) -> TrialDecision:
    required_signals = {"has_type_2_diabetes"}
    exclusion_signals = {
        "kidney_disease_history",
        "creatinine_high",
        "warfarin_interaction",
        "cyp3a4_inhibitor_interaction",
    }
    tag_set = set(reasoning_tags)
    if not required_signals.issubset(tag_set):
        return TrialDecision.INELIGIBLE
    if exclusion_signals.intersection(tag_set):
        return TrialDecision.INELIGIBLE
    if "greater than 18" in " ".join(observation.trial_protocol.inclusion_criteria).lower():
        return TrialDecision.ELIGIBLE if "adult_patient" in tag_set else TrialDecision.INELIGIBLE
    return TrialDecision.ELIGIBLE


def build_prediction(case_id: str) -> tuple[AgentPrediction, float]:
    env = ClinicalTrialEligibilityScreenerEnvironment()
    observation = env.reset(case_id=case_id)
    reasoning_tags = derive_reasoning_tags(observation)
    decision = derive_decision(reasoning_tags, observation)
    thought = (
        "Rule-based baseline reviewed the structured patient profile and protocol. "
        f"Reasoning tags: {', '.join(reasoning_tags)}."
    )
    final_observation = env.step(
        ClinicalTrialAction(
            thought=thought,
            decision=decision,
            metadata={"reasoning_tags": reasoning_tags},
        )
    )
    prediction = AgentPrediction(
        case_id=case_id,
        final_decision=decision,
        reasoning_tags=reasoning_tags,
        steps_taken=env.state.step_count,
        final_reward=final_observation.reward,
        transcript=[
          {
              "decision": decision.value,
              "reward": float(final_observation.reward or 0.0),
              "done": final_observation.done,
          }
        ],
    )
    return prediction, float(final_observation.reward or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=None, help="Run the baseline on a single case id")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    case_ids = [args.case_id] if args.case_id else [case.case_id for case in load_case_definitions()]
    results = []
    for case_id in case_ids:
        prediction, reward = build_prediction(case_id)
        grade = grade_prediction(prediction)
        results.append(
            {
                "case_id": case_id,
                "decision": prediction.final_decision.value,
                "reward": reward,
                "grade_score": grade.total_score,
                "reasoning_tags": prediction.reasoning_tags,
            }
        )

    summary = {
        "cases": results,
        "average_reward": mean(item["reward"] for item in results),
        "average_grade_score": mean(item["grade_score"] for item in results),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    for item in results:
        print(
            f"{item['case_id']}: decision={item['decision']} reward={item['reward']:.3f} "
            f"grade={item['grade_score']:.3f}"
        )
    print(f"average_reward={summary['average_reward']:.3f}")
    print(f"average_grade_score={summary['average_grade_score']:.3f}")


if __name__ == "__main__":
    main()
