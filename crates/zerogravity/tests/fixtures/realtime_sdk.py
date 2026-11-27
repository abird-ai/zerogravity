"""Actual installed SDK against the same cached native model and semantic turns."""
import json
import sys

import httpx
import openai
from openai.types.realtime import RealtimeServerEvent
from pydantic import TypeAdapter


EVENT = TypeAdapter(RealtimeServerEvent)


def receive(connection):
    # The SDK's Realtime parser is unchecked even when HTTP strict validation
    # is enabled. Validate the actual wire bytes with its installed schema.
    raw = connection.recv_bytes()
    EVENT.validate_json(raw, strict=True)
    return connection.parse_event(raw)


def main(address, expected):
    with openai.OpenAI(
        base_url=f"http://{address}/v1",
        api_key="secret",
        max_retries=0,
        http_client=httpx.Client(trust_env=False),
        _strict_response_validation=True,
    ) as client:
        with client.realtime.connect(
            model="alias",
            max_retries=0,
            websocket_connection_options={"open_timeout": 10, "close_timeout": 5, "proxy": None},
        ) as connection:
            event = receive(connection)
            assert event.type == "session.created", event
            assert event.session.model == "alias"
            assert event.session.output_modalities == ["text"]
            connection.session.update(session={"type": "realtime", "output_modalities": ["audio"]})
            event = receive(connection)
            assert event.type == "error" and event.error.code == "unsupported_parameter", event
            connection.session.update(session={
                "type": "realtime",
                "output_modalities": ["text"],
                "instructions": "Answer briefly.",
                "max_output_tokens": 8,
            })
            event = receive(connection)
            assert event.type == "session.updated", event
            for case in expected:
                connection.conversation.item.create(item={
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": case["input"]}],
                })
                assert receive(connection).type == "conversation.item.added"
                assert receive(connection).type == "conversation.item.done"
                connection.response.create()
                response_id = None
                parts = []
                while True:
                    event = receive(connection)
                    assert event.type != "error", event
                    if event.type == "response.created":
                        response_id = event.response.id
                    elif event.type == "response.output_text.delta":
                        assert event.response_id == response_id
                        parts.append(event.delta)
                    elif event.type == "response.output_text.done":
                        assert event.text == "".join(parts)
                    elif event.type == "response.done":
                        response = event.response
                        assert response.id == response_id
                        assert response.status == case["status"], (response, case)
                        assert response.output[0].content[0].text == case["text"] == "".join(parts)
                        assert response.usage.input_tokens == case["input_tokens"]
                        assert response.usage.output_tokens == case["output_tokens"]
                        assert response.usage.total_tokens == case["input_tokens"] + case["output_tokens"]
                        assert response.usage.input_token_details.cached_tokens == case["cached_tokens"]
                        print("OpenAI SDK Realtime native parity:", response.status, repr(case["text"]),
                              response.usage.model_dump(), flush=True)
                        break


if __name__ == "__main__":
    main(sys.argv[1], json.loads(sys.argv[2]))
