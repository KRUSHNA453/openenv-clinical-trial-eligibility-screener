import json

from grader import grade_prediction, load_prediction
from models import AgentPrediction, TrialDecision


def test_grader_returns_full_score_for_correct_submission() -> None:
    prediction = AgentPrediction(
        case_id="hard_drug_interaction",
        final_decision=TrialDecision.INELIGIBLE,
        reasoning_tags=[
            "age_in_range",
            "has_type_2_diabetes",
            "hba1c_in_range",
            "creatinine_within_limit",
            "systolic_bp_within_limit",
            "warfarin_interaction",
            "cyp3a4_inhibitor_interaction",
            "no_stage3_kidney_disease",
            "no_recent_pancreatitis",
        ],
        steps_taken=1,
    )

    report = grade_prediction(prediction)

    assert report.decision_correct is True
    assert report.total_score == 1.0


def test_grader_rejects_non_terminal_decision() -> None:
    prediction = AgentPrediction(
        case_id="medium_ckd_exclusion",
        final_decision=TrialDecision.NEED_MORE_INFO,
        reasoning_tags=["age_in_range", "has_type_2_diabetes"],
        steps_taken=3,
    )

    report = grade_prediction(prediction)

    assert report.terminal_decision_submitted is False
    assert report.decision_score == 0.0


def test_load_prediction_supports_inference_log_files(tmp_path) -> None:
    path = tmp_path / "prediction.log"
    payload = {
        "status": "completed",
        "prediction": {
            "case_id": "easy_t2d_adult",
            "final_decision": "ELIGIBLE",
            "reasoning_tags": ["adult_patient", "has_type_2_diabetes"],
            "steps_taken": 1,
            "final_reward": 0.625,
            "transcript": [],
        },
    }
    path.write_text(f"[END] {json.dumps(payload)}\n", encoding="utf-8")

    prediction = load_prediction(path)

    assert prediction.case_id == "easy_t2d_adult"
    assert prediction.final_decision == TrialDecision.ELIGIBLE
