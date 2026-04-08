from inference import extract_json_payload


def test_extract_json_payload_handles_wrapped_response() -> None:
    payload = extract_json_payload(
        "Sure, here is the result:\n```json\n{\"decision\":\"ELIGIBLE\",\"thought\":\"ok\",\"reasoning_tags\":[]}\n```"
    )

    assert payload["decision"] == "ELIGIBLE"
