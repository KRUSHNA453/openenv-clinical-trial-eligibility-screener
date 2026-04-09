import importlib

import inference as inference_module
from inference import extract_json_payload


def test_extract_json_payload_handles_wrapped_response() -> None:
    payload = extract_json_payload(
        "Sure, here is the result:\n```json\n{\"decision\":\"ELIGIBLE\",\"thought\":\"ok\",\"reasoning_tags\":[]}\n```"
    )

    assert payload["decision"] == "ELIGIBLE"


def test_image_name_alias_populates_local_image_name(monkeypatch) -> None:
    monkeypatch.delenv("LOCAL_IMAGE_NAME", raising=False)
    monkeypatch.setenv("IMAGE_NAME", "validator-image")

    reloaded = importlib.reload(inference_module)

    assert reloaded.LOCAL_IMAGE_NAME == "validator-image"

    monkeypatch.delenv("IMAGE_NAME", raising=False)
    importlib.reload(inference_module)


def test_unknown_task_falls_back_to_all_cases() -> None:
    case_ids = inference_module.resolve_case_ids("validator-task-name")

    assert len(case_ids) == 3
    assert "easy_t2d_adult" in case_ids
