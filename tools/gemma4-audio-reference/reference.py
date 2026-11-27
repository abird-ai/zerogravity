"""Independent CPU F32 Gemma 4 audio-tower + language projection reference.

Only existing local original Safetensors/config files, no AutoModel download,
pickle, generation or claimed speech transcription. The native consumer
authenticates the complete original source separately.
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4TextConfig,
)
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel,
    Gemma4MultimodalEmbedder,
)


def bind(module, source, prefix):
    expected = set(module.state_dict())
    actual = {key.removeprefix(prefix) for key in source.keys() if key.startswith(prefix)}
    if actual != expected:
        raise ValueError(f"original source closure: missing {expected - actual}; extra {actual - expected}")
    module.load_state_dict({key: source.get_tensor(prefix + key) for key in expected}, strict=True)
    return module.eval()


def waveform(length):
    time = np.arange(length, dtype=np.float64) / 16000.0
    envelope = 0.55 + 0.35 * np.sin(2 * np.pi * 5.3 * time)
    return (envelope * (0.22 * np.sin(2 * np.pi * (317 * time + 480 * time**2))
                        + 0.11 * np.sin(2 * np.pi * 1103 * time))).astype(np.float32)


def run(snapshot, output):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    config = json.loads((snapshot / "config.json").read_text())
    processor = json.loads((snapshot / "processor_config.json").read_text())
    audio = Gemma4AudioConfig.from_dict(config["audio_config"])
    text = Gemma4TextConfig.from_dict(config["text_config"])
    # Gemma4AudioAttention consumes a BOOLEAN blocked mask using
    # logical_not, not the additive 0/-inf mask of eager text attention.
    # SDPA here chooses the boolean mask builder; the audio layer still
    # executes its own source-defined blocked attention.
    audio._attn_implementation = "sdpa"
    frontend = Gemma4AudioFeatureExtractor(**processor["feature_extractor"])
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as source:
        tower = bind(Gemma4AudioModel(audio), source, "model.audio_tower.")
        projector = bind(Gemma4MultimodalEmbedder(audio, text), source, "model.embed_audio.")
    cases = []
    def require_boolean_mask(module, args, kwargs):
        mask = kwargs["attention_mask"]
        assert mask.dtype == torch.bool, f"audio mask must be boolean: {mask.dtype}"
        assert mask[0, 0, 0, 0, 12], "first query attends its own key"
        assert not mask[0, 0, 0, 0, 13], "first query cannot attend future key"
    handle = tower.layers[0].self_attn.register_forward_pre_hook(require_boolean_mask, with_kwargs=True)
    with torch.inference_mode():
        for length in (2720, 10240):
            samples = waveform(length)
            features = frontend([samples], sampling_rate=16000, padding=False,
                                pad_to_multiple_of=None, truncation=False, return_tensors="pt")
            mask = features["input_features_mask"].bool()
            assert mask.all(), "single unpadded sequence"
            encoded = tower(features["input_features"], attention_mask=mask, return_dict=True)
            projected = projector(encoded.last_hidden_state)
            assert encoded.attention_mask.all()
            assert torch.isfinite(projected).all()
            # The default processor's right padding is never allowed to
            # change a valid unpadded audio token.
            padded = frontend([samples], sampling_rate=16000, return_tensors="pt",
                              truncation=False)
            padded_result = tower(padded["input_features"], attention_mask=padded["input_features_mask"].bool(),
                                  return_dict=True)
            valid = projector(padded_result.last_hidden_state)[padded_result.attention_mask.bool()]
            torch.testing.assert_close(valid, projected[0], atol=2e-4, rtol=2e-4)
            cases.append({
                "samples": samples.tolist(),
                "frames": features["input_features"].shape[1],
                "features": features["input_features"].flatten().tolist(),
                "tokens": projected.shape[1],
                "width": projected.shape[2],
                "values": projected.flatten().tolist(),
            })
            print(f"REFERENCE samples={length} frames={cases[-1]['frames']} tokens={projected.shape[1]} width={projected.shape[2]} absmax={projected.abs().max().item():.6g}", flush=True)
    handle.remove()
    output.write_text(json.dumps(cases, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    run(args.snapshot, args.output)
