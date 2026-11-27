"""Optional real cached-model canary using the installed official SDK."""
import ipaddress
import sys
import time

import httpx
import openai
from pydantic import BaseModel


def standard(value):
    """The installed SDK permits extra fields even under strict validation."""
    if isinstance(value, BaseModel):
        # _request_id is an SDK-local annotation, not part of the wire object.
        extras = set(value.model_extra or {}) - {"_request_id"}
        # The official Responses wire examples include the boolean store
        # field, although SDK 2.33.0's generated Response class omits it.
        # Keep the established resource semantics, not an SDK schema quirk.
        if type(value).__name__ == "Response" and "store" in extras:
            assert isinstance(value.store, bool)
            extras.remove("store")
        assert not extras, (type(value).__name__, extras)
        for name in type(value).model_fields:
            standard(getattr(value, name, None))
    elif isinstance(value, (list, tuple)):
        for child in value:
            standard(child)


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
    standard(list(models.values()))
    assert client.models.retrieve("fast").model_dump() == models["fast"].model_dump()
    try:
        client.models.retrieve("missing")
        raise AssertionError("unserved model was retrieved")
    except openai.NotFoundError as error:
        assert error.body["param"] == "model" and error.request_id

    # The actual installed SDK must receive a canonical refusal, not an
    # apparently successful dummy audio/image response.
    for call in (
        lambda: client.audio.speech.create(model="fast", input="Hello", voice="alloy"),
        lambda: client.images.generate(model="fast", prompt="Hello"),
    ):
        try:
            call()
            raise AssertionError("unavailable media capability accepted")
        except openai.NotFoundError as error:
            assert error.request_id and error.body["code"] == "not_found"
            assert set(error.body) == {"code", "type", "message", "param"}

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
    chat_kwargs = dict(
        model="served-fixture",
        messages=[
            {"role": "developer", "content": [{"type": "text", "text": instructions}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Continue: "},
                {"type": "text", "text": "1, 2, 3,"},
            ]},
        ],
        max_completion_tokens=2, temperature=0, top_p=1, n=1,
        tools=[], tool_choice="none", parallel_tool_calls=False,
        response_format={"type": "text"}, modalities=["text"], store=False,
        logprobs=False, top_logprobs=0,
    )
    chat = client.chat.completions.create(**chat_kwargs)
    plain = client.chat.completions.create(
        model="served-fixture",
        messages=[{"role": "developer", "content": instructions},
                  {"role": "user", "content": prompt}],
        max_tokens=2, temperature=0,
    )
    assert plain.choices[0].message.content == chat.choices[0].message.content
    assert plain.choices[0].finish_reason == chat.choices[0].finish_reason
    assert plain.usage.model_dump() == chat.usage.model_dump()
    standard([chat, plain])
    with client.chat.completions.create(
        **chat_kwargs, stream=True,
        stream_options={"include_usage": True, "include_obfuscation": False},
    ) as stream:
        chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert all(chunk.id == chunks[0].id for chunk in chunks)
    assert all(chunk.usage is None for chunk in chunks[:-1])
    assert chunks[-1].choices == []
    assert chunks[-1].usage.model_dump() == chat.usage.model_dump()
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks[:-1]) == chat.choices[0].message.content
    assert chunks[-2].choices[0].finish_reason == chat.choices[0].finish_reason
    standard(chunks)
    assert chat.usage.prompt_tokens_details.cached_tokens == 0
    assert chat.usage.completion_tokens_details.reasoning_tokens == 0
    raw_kwargs = dict(model="fast", prompt=prompt, max_tokens=2,
                      temperature=0, top_p=1, n=1, best_of=1, echo=False)
    raw = client.completions.create(**raw_kwargs)
    # SDK 2.33.0's legacy CompletionChoice declares a non-null literal
    # finish_reason even for partial SSE chunks. Its ordinary client mode
    # accepts the actual legacy stream; explicitly validate that boundary.
    with openai.OpenAI(
        api_key="local-fixture", base_url=f"http://{address}/v1",
        max_retries=0, timeout=60,
        http_client=httpx.Client(trust_env=False, timeout=60),
    ) as legacy_client:
        with legacy_client.completions.create(
            **raw_kwargs, stream=True, stream_options={"include_usage": True},
        ) as stream:
            raw_chunks = list(stream)
    assert all(chunk.id == raw_chunks[0].id and chunk.object == "text_completion" for chunk in raw_chunks)
    assert all(chunk.choices[0].finish_reason is None for chunk in raw_chunks[:-2])
    assert raw_chunks[-2].choices[0].finish_reason == raw.choices[0].finish_reason
    assert "".join(chunk.choices[0].text for chunk in raw_chunks if chunk.choices) == raw.choices[0].text
    assert raw_chunks[-1].choices == []
    assert raw_chunks[-1].usage.model_dump() == raw.usage.model_dump()
    standard([raw, *raw_chunks])
    for parameter, value in [
        ("response_format", {"type": "json_object"}),
        ("tools", [{"type": "function", "function": {"name": "lookup"}}]),
        ("stop", ["end"]),
        ("temperature", 0.5),
        ("prompt", "must not be discarded"),
    ]:
        try:
            client.chat.completions.create(
                **{key: value for key, value in chat_kwargs.items() if key != parameter},
                extra_body={parameter: value},
            )
            raise AssertionError(f"unsupported chat parameter accepted: {parameter}")
        except openai.BadRequestError as error:
            assert error.body["param"] == parameter and error.request_id
    try:
        client.completions.create(**{**raw_kwargs, "echo": True})
        raise AssertionError("unsupported legacy echo accepted")
    except openai.BadRequestError as error:
        assert error.body["param"] == "echo" and error.request_id
    print("OpenAI SDK: Chat string/typed text parity, neutral controls, exact JSON/SSE usage, legacy Completions and capability refusals passed")
    counted = client.responses.input_tokens.count(
        model=kwargs["model"], instructions=kwargs["instructions"],
        input=kwargs["input"], truncation="disabled",
    )
    assert counted.object == "response.input_tokens"
    standard(counted)
    response = client.responses.create(**kwargs)
    assert counted.input_tokens == response.usage.input_tokens
    assert response.output_text == chat.choices[0].message.content
    assert response.usage.input_tokens == chat.usage.prompt_tokens
    assert response.usage.output_tokens == chat.usage.completion_tokens
    assert response.usage.total_tokens == chat.usage.total_tokens
    assert response.usage.input_tokens_details.cached_tokens == 0
    assert response.status == ("incomplete" if chat.choices[0].finish_reason == "length" else "completed")
    assert response.store is False and response.metadata == kwargs["metadata"]
    assert response.id == response._request_id.replace("req_", "resp_", 1)
    assert response.output[0].role == "assistant"
    standard(response)
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
    standard(events)
    # Foreground records are explicitly retained, not a second model
    # state. Prior request-local instructions are replaced on continuation.
    stored = client.responses.create(**{**kwargs, "store": True})
    assert stored.store is True
    assert client.responses.retrieve(stored.id).model_dump() == stored.model_dump()
    standard(stored)
    page = client.responses.input_items.list(stored.id, order="asc", limit=1)
    assert len(page.data) == 1 and not page.has_more
    assert page.data[0].role == "user"
    assert page.data[0].content[0].text == prompt
    new_prompt = "Continue."
    new_instructions = "Answer briefly."
    continued_count = client.responses.input_tokens.count(
        model="served-fixture", input=new_prompt, instructions=new_instructions,
        previous_response_id=stored.id,
    )
    continued = client.responses.create(
        model="served-fixture",
        input=new_prompt,
        instructions=new_instructions,
        previous_response_id=stored.id,
        store=True,
        max_output_tokens=2,
        temperature=0,
    )
    expected = client.chat.completions.create(
        model="fast",
        messages=[
            {"role":"developer", "content":new_instructions},
            {"role":"user", "content":prompt},
            {"role":"assistant", "content":stored.output_text},
            {"role":"user", "content":new_prompt},
        ],
        max_tokens=2,
        temperature=0,
    )
    assert continued.previous_response_id == stored.id
    assert continued.output_text == expected.choices[0].message.content
    assert continued.usage.input_tokens == expected.usage.prompt_tokens
    assert continued_count.input_tokens == continued.usage.input_tokens
    assert continued.usage.output_tokens == expected.usage.completion_tokens
    page = client.responses.input_items.list(continued.id, order="asc", limit=2)
    assert len(page.data) == 2 and page.has_more
    assert [item.role for item in page.data] == ["user", "assistant"]
    assert page.data[1].content[0].text == stored.output_text
    tail = client.responses.input_items.list(
        continued.id, order="asc", after=page.last_id, limit=2
    )
    assert len(tail.data) == 1 and not tail.has_more
    assert tail.data[0].content[0].text == new_prompt
    try:
        client.responses.cancel(stored.id)
        raise AssertionError("foreground cancellation was claimed")
    except openai.BadRequestError as error:
        assert error.body["param"] == "background" and error.request_id
    # SDK 2.33.0 declares delete() -> None. Verify its actual HTTP
    # response, then confirm the resource is no longer retrievable.
    deleted = client.responses.with_raw_response.delete(stored.id)
    assert deleted.status_code == 200
    assert deleted.http_response.json() == {
        "id": stored.id, "object": "response.deleted", "deleted": True,
    }
    try:
        client.responses.retrieve(stored.id)
        raise AssertionError("deleted response remained available")
    except openai.NotFoundError:
        pass
    assert client.responses.retrieve(continued.id).model_dump() == continued.model_dump()
    try:
        client.responses.retrieve(response.id)
        raise AssertionError("stateless response was retained")
    except openai.NotFoundError:
        pass
    try:
        client.responses.create(
            model="fast", input=prompt, previous_response_id=stored.id, store=False
        )
        raise AssertionError("deleted continuation was accepted")
    except openai.NotFoundError as error:
        assert error.body["param"] == "previous_response_id"
    assert client.responses.delete(continued.id) is None
    try:
        client.responses.retrieve(continued.id)
        raise AssertionError("deleted child remained available")
    except openai.NotFoundError:
        pass
    # The create HTTP receipt is no longer the background execution owner.
    # Poll the exact ID through the installed official client.
    for retain in (True, False):
        job = client.responses.create(**{**kwargs, "background": True, "store": retain})
        assert job.background is True and job.store is retain
        assert job.status == "in_progress" and job.usage is None
        deadline = time.monotonic() + 60
        while job.status in ("queued", "in_progress"):
            assert time.monotonic() < deadline, "background deadline"
            time.sleep(0.05)
            job = client.responses.retrieve(job.id)
        assert job.status == response.status
        assert job.output_text == response.output_text
        assert job.usage == response.usage
        assert client.responses.cancel(job.id).model_dump() == job.model_dump()
        if retain:
            child = client.responses.create(
                model="fast", input=new_prompt, instructions=new_instructions,
                previous_response_id=job.id, store=False, max_output_tokens=2,
                temperature=0,
            )
            assert child.output_text == expected.choices[0].message.content
            assert child.usage.input_tokens == expected.usage.prompt_tokens
        else:
            try:
                client.responses.create(
                    model="fast", input=new_prompt, previous_response_id=job.id,
                    store=False,
                )
                raise AssertionError("temporary polling record accepted continuation")
            except openai.BadRequestError as error:
                assert error.body["param"] == "previous_response_id"
        assert client.responses.delete(job.id) is None

    # A real small model can finish before the next HTTP request arrives.
    # Either cancellation wins, or the already published terminal wins;
    # the deterministic held-receipt socket test proves the in-flight case.
    job = client.responses.create(**{**kwargs, "background": True})
    stopped = client.responses.cancel(job.id)
    assert stopped.status in ("cancelled", response.status)
    assert stopped.background is True and stopped.id == job.id
    if stopped.status == "cancelled":
        assert stopped.usage is None
    else:
        assert stopped.output_text == response.output_text
        assert stopped.usage == response.usage
    assert client.responses.cancel(job.id).model_dump() == stopped.model_dump()
    assert client.responses.retrieve(job.id).model_dump() == stopped.model_dump()
    assert client.responses.delete(job.id) is None
    print(f"OpenAI SDK {openai.__version__}: Models, Responses JSON/SSE, exact Chat parity, stored response/continuation, background polling and idempotent cancel ({stopped.status}) passed")

    # Conversations are charged protocol context, not a retained model session.
    conversation = client.conversations.create(
        metadata={"canary": "oa4c"},
        items=[{"role": "user", "content": prompt}],
    )
    cid = conversation.id
    assert client.conversations.retrieve(cid).model_dump() == conversation.model_dump()
    updated = client.conversations.update(cid, metadata={"canary": "updated"})
    assert updated.created_at == conversation.created_at
    assert updated.metadata == {"canary": "updated"}
    standard([conversation, updated])
    initial = client.conversations.items.list(cid, order="asc")
    assert len(initial.data) == 1 and initial.data[0].content[0].text == prompt
    standard(initial.data)
    initial_id = initial.data[0].id
    # Omitted input replays the original stored typed history, without
    # re-appending that history or persisting request-local instructions.
    first_count = client.responses.input_tokens.count(
        model="fast", conversation=cid, instructions=instructions,
    )
    assert client.conversations.items.list(cid, order="asc").model_dump() == initial.model_dump()
    first = client.responses.create(
        model="fast", conversation=cid, instructions=instructions,
        store=False, max_output_tokens=2, temperature=0,
    )
    assert first.conversation.id == cid
    assert first.output_text == response.output_text
    assert first.usage == response.usage
    assert first_count.input_tokens == first.usage.input_tokens
    first_page = client.conversations.items.list(cid, order="asc")
    assert len(first_page.data) == 2
    assert first_page.data[0].id == initial_id
    assert first_page.data[1].id == first.output[0].id
    assert first_page.data[1].content[0].text == first.output_text
    try:
        client.responses.retrieve(first.id)
        raise AssertionError("stateless conversation response was retained")
    except openai.NotFoundError:
        pass

    next_count = client.responses.input_tokens.count(
        model="served-fixture", conversation={"id": cid}, input=new_prompt,
        instructions=new_instructions,
    )
    assert client.conversations.items.list(cid, order="asc").model_dump() == first_page.model_dump()
    with client.responses.create(
        model="served-fixture", conversation={"id": cid}, input=new_prompt,
        instructions=new_instructions, store=True, stream=True,
        max_output_tokens=2, temperature=0,
    ) as stream:
        conversation_events = list(stream)
    last = conversation_events[-1].response
    assert last.conversation.id == cid
    assert last.output_text == expected.choices[0].message.content
    assert last.usage.input_tokens == expected.usage.prompt_tokens
    assert next_count.input_tokens == last.usage.input_tokens
    standard(conversation_events)
    assert last.usage.output_tokens == expected.usage.completion_tokens
    page = client.conversations.items.list(cid, order="asc", limit=2)
    assert page.has_more and len(page.data) == 2
    tail = client.conversations.items.list(cid, order="asc", after=page.last_id)
    assert not tail.has_more and len(tail.data) == 2
    assert tail.data[0].content[0].text == new_prompt
    assert tail.data[1].id == last.output[0].id
    assert tail.data[1].content[0].text == last.output_text
    standard([*page.data, *tail.data])
    all_input = client.responses.input_items.list(last.id, order="asc")
    assert [item.id for item in all_input.data] == [
        initial_id, first.output[0].id, tail.data[0].id,
    ]
    added = client.conversations.items.create(
        cid, items=[{"role": "user", "content": "Temporary edit"}],
    )
    added_id = added.data[0].id
    assert client.conversations.items.retrieve(
        added_id, conversation_id=cid
    ).model_dump() == added.data[0].model_dump()
    assert client.conversations.items.delete(
        added_id, conversation_id=cid
    ).id == cid
    try:
        client.conversations.items.retrieve(added_id, conversation_id=cid)
        raise AssertionError("deleted conversation item remained available")
    except openai.NotFoundError:
        pass
    deleted = client.conversations.delete(cid)
    assert deleted.deleted and deleted.id == cid
    try:
        client.conversations.retrieve(cid)
        raise AssertionError("deleted conversation remained available")
    except openai.NotFoundError:
        pass
    assert client.responses.retrieve(last.id).model_dump() == last.model_dump()
    assert client.responses.delete(last.id) is None

    conversation = client.conversations.create()
    job = client.responses.create(
        **kwargs, conversation=conversation.id, background=True,
    )
    deadline = time.monotonic() + 60
    while job.status in ("queued", "in_progress"):
        assert time.monotonic() < deadline, "conversation background deadline"
        time.sleep(0.05)
        job = client.responses.retrieve(job.id)
    assert job.status == response.status
    assert job.output_text == response.output_text
    assert job.usage == response.usage
    page = client.conversations.items.list(conversation.id, order="asc")
    assert len(page.data) == 2 and page.data[0].content[0].text == prompt
    assert page.data[1].id == job.output[0].id
    assert client.responses.delete(job.id) is None
    assert client.conversations.delete(conversation.id).deleted
    print("OpenAI SDK: conversation CRUD, stable item cursors, exact Chat parity, foreground/SSE/background atomic append and independent response retention passed")

    for params, error_type, parameter in [
        ({"model":"missing","input":prompt}, openai.NotFoundError, "model"),
        ({"model":"fast","input":prompt,"conversation":"conv_missing"}, openai.NotFoundError, "conversation"),
        ({"model":"fast","input":prompt,"previous_response_id":stored.id}, openai.NotFoundError, "previous_response_id"),
        ({"model":"fast","input":prompt,"truncation":"auto"}, openai.BadRequestError, "truncation"),
        ({"model":"fast","input":prompt,"reasoning":{"effort":"high"}}, openai.BadRequestError, "reasoning"),
    ]:
        try:
            client.responses.input_tokens.count(**params)
            raise AssertionError("invalid token-count request was accepted")
        except error_type as error:
            assert error.body["param"] == parameter and error.request_id
    print("OpenAI SDK: exact input-token count matches generation usage for typed text, previous-response replay and read-only conversation snapshots")

    # Resume the exact event log from a disconnected creation stream. This
    # uses the ordinary SDK, not a ZeroGravity client or a second request.
    for retain in (True, False):
        with client.responses.create(**{**kwargs, "background": True, "stream": True, "store": retain}) as stream:
            created = next(stream)
            assert created.type == "response.created"
            assert created.sequence_number == 0
            bg_id = created.response.id
            assert created.response.background is True
        with client.responses.retrieve(bg_id, stream=True, starting_after=created.sequence_number, include_obfuscation=False) as stream:
            resumed = list(stream)
        events = [created, *resumed]
        assert [event.sequence_number for event in events] == list(range(len(events)))
        final = events[-1].response
        assert final.status == response.status
        assert final.output_text == response.output_text
        assert final.usage == response.usage
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == final.output_text
        assert client.responses.retrieve(bg_id).model_dump() == final.model_dump()
        with client.responses.retrieve(bg_id, stream=True) as stream:
            assert [event.model_dump() for event in stream] == [event.model_dump() for event in events]
        with client.responses.retrieve(bg_id, stream=True, starting_after=events[-1].sequence_number) as stream:
            assert list(stream) == []
        assert client.responses.cancel(bg_id).model_dump() == final.model_dump()
        assert client.responses.delete(bg_id) is None

    # Conversation commit is independent of the creating and replaying
    # sockets, and happens exactly once at successful terminal publication.
    conversation = client.conversations.create()
    with client.responses.create(**{**kwargs, "background": True, "stream": True, "conversation": conversation.id}) as stream:
        created = next(stream)
        bg_id = created.response.id
    with client.responses.retrieve(bg_id, stream=True, starting_after=created.sequence_number) as stream:
        events = list(stream)
    final = events[-1].response
    assert final.output_text == response.output_text and final.usage == response.usage
    page = client.conversations.items.list(conversation.id, order="asc")
    assert len(page.data) == 2 and page.data[1].id == final.output[0].id
    with client.responses.retrieve(bg_id, stream=True) as stream:
        assert list(stream)[-1].response.model_dump() == final.model_dump()
    assert client.conversations.items.list(conversation.id, order="asc").model_dump() == page.model_dump()
    assert client.responses.delete(bg_id) is None
    assert client.conversations.delete(conversation.id).deleted
    print("OpenAI SDK: background streaming, create disconnect/resume, exact full replay, terminal usage and once-only conversation append passed")

    with client.responses.create(**{**kwargs, "background": True, "stream": True}) as stream:
        created = next(stream)
        bg_id = created.response.id
    stopped_stream = client.responses.cancel(bg_id)
    with client.responses.retrieve(bg_id, stream=True, starting_after=created.sequence_number) as stream:
        tail = list(stream)
    assert [e.sequence_number for e in tail] == list(range(1, len(tail) + 1))
    if stopped_stream.status == "cancelled":
        # The SSE event union has no cancelled terminal; the canonical
        # status is available from cancellation and JSON polling.
        assert all(e.type not in ("response.completed", "response.incomplete", "response.failed") for e in tail)
    else:
        assert tail[-1].response.model_dump() == stopped_stream.model_dump()
    assert client.responses.cancel(bg_id).model_dump() == stopped_stream.model_dump()
    assert client.responses.delete(bg_id) is None
    print(f"OpenAI SDK: background streaming cancellation race ({stopped_stream.status}) passed")

print("OpenAI SDK: ordinary text, model, stream and resource objects contain no undeclared fields; unavailable media refused")
