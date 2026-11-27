"""Local mono PCM speech: independent original Gemma 4 F32 greedy transcript."""
import argparse
import json
import pathlib
import wave
import gc
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder
from transformers.utils.chat_template_utils import render_jinja_template
from reference import bind
from language import load_text


def run(snapshot, wav, output):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    config = json.loads((snapshot / "config.json").read_text())
    processor = json.loads((snapshot / "processor_config.json").read_text())
    with wave.open(str(wav), "rb") as f:
        assert (f.getnchannels(), f.getframerate(), f.getsampwidth()) == (1, 16000, 2)
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.
    audio = Gemma4AudioConfig.from_dict(config["audio_config"])
    text = Gemma4TextConfig.from_dict(config["text_config"])
    audio._attn_implementation = "sdpa"
    frontend = Gemma4AudioFeatureExtractor(**processor["feature_extractor"])
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as source:
        tower = bind(Gemma4AudioModel(audio), source, "model.audio_tower.")
        projector = bind(Gemma4MultimodalEmbedder(audio, text), source, "model.embed_audio.")
    with torch.inference_mode():
        features = frontend([samples], sampling_rate=16000, return_tensors="pt", truncation=False)
        encoded = tower(features["input_features"], attention_mask=features["input_features_mask"].bool(), return_dict=True)
        rows = projector(encoded.last_hidden_state)[encoded.attention_mask.bool()]
    del tower, projector, encoded, features
    gc.collect()
    template = (snapshot / "chat_template.jinja").read_text()
    rendered, _ = render_jinja_template(conversations=[[{"role":"user","content":[{"type":"text","text":"Transcribe the following speech segment in its original language."},{"type":"audio"}]}]], chat_template=template, add_generation_prompt=True, bos_token="<bos>", enable_thinking=False)
    rendered = rendered[0].replace("<|audio|>", "<|audio>" + "<|audio|>" * rows.shape[0] + "<audio|>")
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    print("REFERENCE speech samples",len(samples),"rows",rows.shape[0],"prompt",ids, flush=True)
    model = load_text(snapshot, text)
    with torch.inference_mode():
        input_ids = torch.tensor([ids],dtype=torch.long)
        mask = input_ids == config["audio_token_id"]
        text_ids = input_ids.masked_fill(mask,text.pad_token_id)
        embedded = model.embed_tokens(text_ids)
        ple = model.get_per_layer_inputs(text_ids, torch.where(mask[...,None],model.embed_tokens.weight[text.pad_token_id],embedded))
        embedded = embedded.masked_scatter(mask[...,None].expand_as(embedded),rows)
        result = model(inputs_embeds=embedded, per_layer_inputs=ple, use_cache=True, return_dict=True)
        generated = []
        finish = "length"
        for i in range(8):
            logits = torch.nn.functional.linear(result.last_hidden_state[:,-1,:],model.embed_tokens.weight)
            cap = text.final_logit_softcapping
            if cap:
                logits = (logits / cap).tanh() * cap
            assert torch.isfinite(logits).all()
            token = int(logits.argmax())
            generated.append(token)
            if token in (1,106,50):
                finish = "stop"
                break
            if i < 7:
                result = model(input_ids=torch.tensor([[token]]),past_key_values=result.past_key_values,use_cache=True,return_dict=True)
        transcript = tokenizer.decode(generated[:-1] if finish=="stop" else generated, skip_special_tokens=True)
        print("REFERENCE transcript",repr(transcript),"ids",generated,"finish",finish,flush=True)
        output.write_text(json.dumps(dict(samples=samples.tolist(), prompt=ids, tokens=generated,text=transcript,finish=finish),allow_nan=False))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=pathlib.Path)
    parser.add_argument("wav",type=pathlib.Path)
    parser.add_argument("output",type=pathlib.Path)
    args=parser.parse_args()
    run(args.snapshot,args.wav,args.output)
