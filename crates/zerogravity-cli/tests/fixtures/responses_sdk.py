"""Optional real cached-model canary using the installed official SDK."""
import ipaddress
import sys

import httpx
import openai

address = sys.argv[1]
host, port = address.rsplit(":", 1)
assert ipaddress.ip_address(host.strip("[]")).is_loopback
assert 0 < int(port) < 65536

with openai.OpenAI(
    api_key="local-fixture",
    base_url=f"http://{address}/v1",
    max_retries=0,
    timeout=60,
    http_client=httpx.Client(trust_env=False, timeout=60),
    _strict_response_validation=True,
) as client:
    models = {model.id: model for model in client.models.list().data}
    assert set(models) == {"fast", "served-fixture"}, models
    assert client.models.retrieve("fast").model_dump() == models["fast"].model_dump()
    try:
        client.models.retrieve("missing")
        raise AssertionError("unserved model was retrieved")
    except openai.NotFoundError as error:
        assert error.body["param"] == "model" and error.request_id

    # Responses instructions and text parts must yield the exact same
    # selected-model input as the existing typed Chat request.
    instructions = "Brief."
    prompt = "Continue: 1, 2, 3,"
    kwargs = dict(
        model="fast",
        instructions=instructions,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": "Continue: "},
            {"type": "input_text", "text": "1, 2, 3,"},
        ]}],
        store=False,
        temperature=0,
        max_output_tokens=2,
        metadata={"canary": "oa3"},
    )
    chat = client.chat.completions.create(
        model="served-fixture",
        messages=[{"role": "developer", "content": instructions},
                  {"role": "user", "content": prompt}],
        max_tokens=2,
        temperature=0,
    )
    response = client.responses.create(**kwargs)
    assert response.output_text == chat.choices[0].message.content
    assert response.usage.input_tokens == chat.usage.prompt_tokens
    assert response.usage.output_tokens == chat.usage.completion_tokens
    assert response.usage.total_tokens == chat.usage.total_tokens
    assert response.usage.input_tokens_details.cached_tokens == 0
    assert response.status == ("incomplete" if chat.choices[0].finish_reason == "length" else "completed")
    assert response.store is False and response.metadata == kwargs["metadata"]
    assert response.id == response._request_id.replace("req_", "resp_", 1)
    assert response.output[0].role == "assistant"
    if response.status == "incomplete":
        assert response.incomplete_details.reason == "max_output_tokens"
    with client.responses.create(**kwargs, stream=True) as stream:
        events = list(stream)
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert [event.type for event in events[:4]] == [
        "response.created", "response.in_progress",
        "response.output_item.added", "response.content_part.added",
    ]
    deltas = [event for event in events if event.type == "response.output_text.delta"]
    assert "".join(event.delta for event in deltas) == response.output_text
    assert events[-4].type == "response.output_text.done"
    assert events[-3].type == "response.content_part.done"
    assert events[-2].type == "response.output_item.done"
    final = events[-1].response
    assert events[-1].type == f"response.{response.status}"
    assert final.status == response.status
    assert final.output_text == response.output_text
    assert final.usage == response.usage
    assert final.id != response.id
    assert all(event.item_id == final.output[0].id for event in deltas)
    assert events[0].response.id == final.id
    try:
        client.responses.create(model="fast", input=prompt, store=True)
        raise AssertionError("unimplemented store was accepted")
    except openai.BadRequestError as error:
        assert error.body["param"] == "store" and error.request_id
    print(f"OpenAI SDK {openai.__version__}: Models, Responses JSON/SSE, exact Chat parity, refusal and correlation passed")
