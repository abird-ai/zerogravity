"""Actual original Gemma4 F32 audio-conditioned decoder reference.

Input: prior independent audio reference JSON (real source-derived rows),
local original snapshot only. Bind exact text tensor closure. Evaluate
original text model with the multimodal scatter/PLE contract, and then
one genuine cached continuation. No model or SDK downloads.
"""
import argparse
import json
import pathlib
import gc

import torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
)
from tokenizers import Tokenizer


def load_text(snapshot, config):
    config.dtype = torch.float32
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Gemma4TextModel(config)
    prefix = "model.language_model."
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as source:
        expected = set(model.state_dict())
        actual = {key.removeprefix(prefix) for key in source.keys() if key.startswith(prefix)}
        # The official checkpoint retains unused K/V projections/norms for
        # shared-KV consumers. The actual reference layer constructors omit
        # them. Accept only these explicitly identified original members.
        unused = {
            f"layers.{i}.self_attn.{name}.weight"
            for i, layer in enumerate(model.layers)
            if layer.self_attn.is_kv_shared_layer
            for name in ("k_proj", "v_proj", "k_norm")
        }
        if expected - actual or actual - expected - unused:
            raise ValueError(f"text source closure: missing {expected - actual}; extra {actual - expected - unused}")
        # Assign rather than initializing or copying another full model.
        # BF16 source coefficient -> exact F32 widening.
        state = {}
        for name in sorted(expected):
            weight = source.get_tensor(prefix + name)
            assert weight.dtype == torch.bfloat16
            state[name] = weight.to(dtype=torch.float32)
        model.load_state_dict(state, strict=True, assign=True)
        del state
    # Nonpersistent buffers are derived by the original model classes;
    # meta construction did not initialize their actual CPU storage.
    model.rotary_emb = Gemma4TextRotaryEmbedding(config, device="cpu")
    for embed in (model.embed_tokens, model.embed_tokens_per_layer):
        embed.embed_scale = torch.tensor(embed.scalar_embed_scale, dtype=torch.float32)
    assert all(not t.is_meta for t in model.parameters())
    assert all(not t.is_meta for t in model.buffers())
    gc.collect()
    return model.eval()


def run(snapshot, audio_reference, output):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    config = json.loads((snapshot / "config.json").read_text())
    c = Gemma4TextConfig.from_dict(config["text_config"])
    audio = json.loads(audio_reference.read_text())[0]
    audio_rows = torch.tensor(audio["values"], dtype=torch.float32).reshape(1, audio["tokens"], audio["width"])
    assert audio_rows.shape[-1] == c.hidden_size
    ids = [c.bos_token_id, config["boa_token_id"]] + [config["audio_token_id"]] * audio["tokens"] + [config["eoa_token_id"]]
    # A compact original-token prompt; no assertion that it is an ASR chat
    # template or that a synthetic tone has a meaningful transcript.
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    ids.extend(tokenizer.encode(" Sound:", add_special_tokens=False).ids)
    del tokenizer
    print(f"REFERENCE prompt={ids}, audio_rows={audio_rows.shape[1]}", flush=True)
    model = load_text(snapshot, c)
    print("REFERENCE exact F32 text weights and source-derived buffers loaded", flush=True)
    def logits(hidden):
        result = torch.nn.functional.linear(hidden[:, -1, :], model.embed_tokens.weight)
        cap = c.final_logit_softcapping
        return (result / cap).tanh() * cap if cap else result
    with torch.inference_mode():
        input_ids = torch.tensor([ids], dtype=torch.long)
        mask = input_ids == config["audio_token_id"]
        text_ids = input_ids.masked_fill(mask, c.pad_token_id)
        embedded = model.embed_tokens(text_ids)
        # Original Gemma4Model.forward: PLE lookup before media scatter.
        # get_per_layer_inputs uses IDs when present; subsequent learned
        # projection uses the merged main embedding, not the PAD embedding.
        per_layer = model.get_per_layer_inputs(text_ids, torch.where(mask[..., None], model.embed_tokens.weight[c.pad_token_id], embedded))
        embedded = embedded.masked_scatter(mask[..., None].expand_as(embedded), audio_rows)
        first = model(inputs_embeds=embedded, per_layer_inputs=per_layer, use_cache=True, return_dict=True)
        prefill = logits(first.last_hidden_state)
        assert torch.isfinite(prefill).all()
        token = int(prefill.argmax())
        print(f"REFERENCE prefill vocabulary={prefill.numel()} argmax={token}", flush=True)
        second = model(input_ids=torch.tensor([[token]]), past_key_values=first.past_key_values,
                       use_cache=True, return_dict=True)
        decode = logits(second.last_hidden_state)
        assert torch.isfinite(decode).all()
        print(f"REFERENCE cached continuation argmax={int(decode.argmax())}", flush=True)
        output.write_text(json.dumps({"prompt": ids, "audio_case": 0, "decode_token": token,
                                      "prefill": prefill.flatten().tolist(),
                                      "decode": decode.flatten().tolist()}, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=pathlib.Path)
    parser.add_argument("audio_reference", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    run(args.snapshot, args.audio_reference, args.output)
