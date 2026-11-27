"""Actual installed SDK against the source-bound synthetic encoder fixture."""
import base64
import struct
import sys

import httpx
import openai

def main(address):
    options = dict(
        api_key="secret", base_url=f"http://{address}/v1",
        max_retries=0, timeout=10,
    )
    # Strict SDK schema for ordinary float response.
    with openai.OpenAI(
        **options, _strict_response_validation=True,
        http_client=httpx.Client(trust_env=False),
    ) as strict:
        assert {model.id for model in strict.models.list().data} == {"fixture", "public-alias"}
        floats = strict.embeddings.create(model="fixture", input=["HÉLLOS", "hello"], encoding_format="float")
        assert floats.object == "list"
        assert [row.index for row in floats.data] == [0, 1]
        assert floats.usage.prompt_tokens == floats.usage.total_tokens == 7
        explicit_ids = strict.embeddings.create(model="public-alias", input=[[2, 5, 6, 3], [2, 5, 3]], encoding_format="float")
        assert [row.embedding for row in explicit_ids.data] == [row.embedding for row in floats.data]
        for params, parameter in [
            (dict(dimensions=2), "dimensions"),
            (dict(input=[7]), "input"),
        ]:
            try:
                strict.embeddings.create(**({"model": "fixture", "input": "hello"} | params))
                raise AssertionError("unsupported embedding request accepted")
            except openai.BadRequestError as error:
                assert error.body["param"] == parameter and error.request_id
        try:
            strict.embeddings.create(model="private", input="hello")
            raise AssertionError("native-only candidate exposed")
        except openai.NotFoundError:
            pass
    # The SDK's Embedding.embedding schema is list[float], even for an
    # explicitly requested base64 wire string. Its ordinary mode applies
    # the official default base64 post-parser; verify that result exactly.
    with openai.OpenAI(**options, http_client=httpx.Client(trust_env=False)) as ordinary:
        decoded = ordinary.embeddings.create(model="fixture", input=["HÉLLOS", "hello"])
        # JSON's shortest f32 decimal is parsed as a Python double.
        # Compare original f32 bits, not the intermediate double rounding.
        def bits(row):
            return struct.pack("<4f", *row.embedding)
        assert [bits(row) for row in decoded.data] == [bits(row) for row in floats.data]
        assert decoded.usage.model_dump() == floats.usage.model_dump()
        encoded = ordinary.embeddings.create(model="fixture", input="HÉLLOS", encoding_format="base64")
        assert isinstance(encoded.data[0].embedding, str)
        raw = base64.b64decode(encoded.data[0].embedding, validate=True)
        assert raw == bits(floats.data[0])
    print("OpenAI SDK: embedding model visibility, float/ID/base64 parity, exact usage and refusal passed")

if __name__ == "__main__":
    main(sys.argv[1])
