"""Installed official OpenAI SDK against actual cached learned speech.

No acquisition, reference transcript injection or HTTP-side inference.
"""
import pathlib
import sys
import json

import httpx
import openai


def main(address, wav, reference_path, task="transcription"):
    reference = json.loads(pathlib.Path(reference_path).read_text())
    with openai.OpenAI(
        base_url=f"http://{address}/v1", api_key="secret", max_retries=0,
        timeout=570, _strict_response_validation=True,
        http_client=httpx.Client(trust_env=False, timeout=570),
    ) as client:
        assert {m.id for m in client.models.list().data} == {"speech", "speech-alias"}
        assert client.models.retrieve("speech-alias").id == "speech-alias"
        audio = pathlib.Path(wav).read_bytes()
        if task == "translation":
            for model, extra, error_type in [
                ("private", {}, openai.NotFoundError),
                ("speech", {"prompt": "unsupported custom prompt"}, openai.BadRequestError),
                ("speech", {"temperature": 0.5}, openai.BadRequestError),
                ("speech", {"response_format": "verbose_json"}, openai.BadRequestError),
            ]:
                try:
                    client.audio.translations.create(model=model, file=("speech.wav", audio, "audio/wav"), **extra)
                    raise AssertionError("unsupported translation capability admitted")
                except error_type as error:
                    assert error.request_id
            reply = client.audio.translations.create(
                model="speech-alias", file=("speech.wav", audio, "audio/wav"),
                response_format="json", temperature=0.0,
            )
            source, target = reference["text"].split("\nEnglish: ")
            assert source == "Hola mundo" and target == "Hello world", reference
            assert reply.model_dump() == {"text": target}, reply
            print("OpenAI SDK learned translation PASS:", reply.model_dump(), flush=True)
            return
        for model, extra, error_type in [
            ("private", {}, openai.NotFoundError),
            ("speech", {"language": "en"}, openai.BadRequestError),
            ("speech", {"stream": True}, openai.BadRequestError),
            ("speech", {"temperature": 0.5}, openai.BadRequestError),
        ]:
            try:
                client.audio.transcriptions.create(model=model, file=("speech.wav", audio, "audio/wav"), **extra)
                raise AssertionError("unsupported capability admitted")
            except error_type as error:
                assert error.request_id
        reply = client.audio.transcriptions.create(
            model="speech-alias", file=("speech.wav", audio, "audio/wav"),
            response_format="json", temperature=0.0, stream=False,
        )
        assert reply.text == reference["text"] == "Hello world", reply
        assert reply.usage.type == "tokens"
        assert reply.usage.input_tokens == len(reference["prompt"])
        assert reply.usage.output_tokens == len(reference["tokens"])
        assert reply.usage.total_tokens == len(reference["prompt"]) + len(reference["tokens"])
        print("OpenAI SDK learned transcription PASS:", reply.model_dump(), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
