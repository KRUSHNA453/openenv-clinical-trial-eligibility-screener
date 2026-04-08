"""Deterministic grader for clinical trial screening submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models import AgentPrediction, GradeReport, TrialDecision
from server.case_data import load_ground_truth_index


TERMINAL_DECISIONS = {TrialDecision.ELIGIBLE, TrialDecision.INELIGIBLE}


def _normalized_tags(tags: list[str]) -> list[str]:
    return sorted({tag.strip().lower() for tag in tags if tag and tag.strip()})


def load_prediction(prediction_path: Path) -> AgentPrediction:
    text = prediction_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prediction file is empty: {prediction_path}")

    if text.startswith("{"):
        payload: dict[str, Any] = json.loads(text)
    else:
        end_lines = [line for line in text.splitlines() if line.startswith("[END] ")]
        if not end_lines:
            raise ValueError(
                "Prediction file must be JSON or contain an [END] JSON line from inference.py"
            )
        payload = json.loads(end_lines[-1][len("[END] ") :])

    record = payload.get("prediction", payload)
    normalized = {
        "case_id": record["case_id"],
        "final_decision": record.get("final_decision", record.get("decision")),
        "reasoning_tags": _normalized_tags(record.get("reasoning_tags", [])),
        "steps_taken": record.get("steps_taken", 0),
        "final_reward": record.get("final_reward"),
        "transcript": record.get("transcript", []),
    }
    return AgentPrediction.model_validate(normalized)


def grade_prediction(prediction: AgentPrediction) -> GradeReport:
    ground_truth = load_ground_truth_index()[prediction.case_id]
    submitted_tags = set(_normalized_tags(prediction.reasoning_tags))
    expected_tags = set(_normalized_tags(ground_truth.reasoning_tags))
    matched_tags = sorted(submitted_tags.intersection(expected_tags))
    missing_tags = sorted(expected_tags.difference(submitted_tags))
    unexpected_tags = sorted(submitted_tags.difference(expected_tags))

    terminal = prediction.final_decision in TERMINAL_DECISIONS
    predicted_eligible = None if not terminal else prediction.final_decision == TrialDecision.ELIGIBLE
    decision_correct = terminal and predicted_eligible == ground_truth.eligible
    reasoning_precision = len(matched_tags) / len(submitted_tags) if submitted_tags else 0.0
    reasoning_recall = len(matched_tags) / len(expected_tags) if expected_tags else 1.0
    if reasoning_precision + reasoning_recall:
        reasoning_f1 = (2 * reasoning_precision * reasoning_recall) / (
            reasoning_precision + reasoning_recall
        )
    else:
        reasoning_f1 = 0.0
    decision_score = 1.0 if decision_correct else 0.0
    total_score = (0.5 * decision_score) + (0.5 * reasoning_f1)

    return GradeReport(
        case_id=prediction.case_id,
        expected_eligible=ground_truth.eligible,
        predicted_eligible=predicted_eligible,
        terminal_decision_submitted=terminal,
        decision_correct=decision_correct,
        expected_reasoning_tags=sorted(expected_tags),
        matched_reasoning_tags=matched_tags,
        missing_reasoning_tags=missing_tags,
        unexpected_reasoning_tags=unexpected_tags,
        reasoning_tag_precision=reasoning_precision,
        reasoning_tag_recall=reasoning_recall,
        reasoning_tag_f1=reasoning_f1,
        decision_score=decision_score,
        total_score=total_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_file", type=Path, help="Path to a JSON prediction file or logs")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the grading report as JSON",
    )
    args = parser.parse_args()

    report = grade_prediction(load_prediction(args.prediction_file))
    report_json = json.dumps(report.model_dump(), indent=2)
    print(report_json)

    if args.output is not None:
        args.output.write_text(report_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
