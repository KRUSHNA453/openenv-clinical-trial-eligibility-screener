import pytest

from models import ClinicalTrialAction, TrialDecision
from server.env import ClinicalTrialEligibilityScreenerEnvironment


def test_reset_loads_requested_case() -> None:
    env = ClinicalTrialEligibilityScreenerEnvironment()
    observation = env.reset(case_id="easy_t2d_adult")

    assert observation.case_id == "easy_t2d_adult"
    assert observation.patient_profile.age == 52
    assert env.state.step_count == 0


def test_partial_credit_reward_is_granted() -> None:
    env = ClinicalTrialEligibilityScreenerEnvironment()
    env.reset(case_id="medium_ckd_exclusion")

    observation = env.step(
        ClinicalTrialAction(
            thought="Type 2 diabetes with low HbA1c #has_type_2_diabetes #hba1c_under_7",
            decision=TrialDecision.PENDING,
        )
    )

    assert observation.reward == pytest.approx(0.3)
    assert observation.reward_breakdown.correct_reasoning_tags == [
        "has_type_2_diabetes",
        "hba1c_under_7",
    ]


def test_hallucination_penalty_is_applied() -> None:
    env = ClinicalTrialEligibilityScreenerEnvironment()
    env.reset(case_id="easy_t2d_adult")

    observation = env.step(
        ClinicalTrialAction(
            thought=(
                "Adult patient with diabetes but patient is pregnant and taking warfarin "
                "#adult_patient #has_type_2_diabetes"
            ),
            decision=TrialDecision.ELIGIBLE,
        )
    )

    assert len(observation.reward_breakdown.hallucinated_claims) == 2
    assert observation.reward_breakdown.score < 0.625


def test_step_penalty_kicks_in_after_ten_steps() -> None:
    env = ClinicalTrialEligibilityScreenerEnvironment()
    env.reset(case_id="easy_t2d_adult")

    observation = None
    for _ in range(11):
        observation = env.step(
            ClinicalTrialAction(
                thought="Still reviewing #adult_patient #has_type_2_diabetes",
                decision=TrialDecision.PENDING,
            )
        )

    assert observation is not None
    assert observation.reward_breakdown.step_penalty == 0.03
