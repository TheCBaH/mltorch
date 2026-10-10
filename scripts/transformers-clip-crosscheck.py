#!/usr/bin/env python3
"""Check bin/transformers_clip_demo against the reference stack.

Usage: transformers-clip-crosscheck.py ASSET_DIR SAFETENSORS IMAGE.ppm ENGINE_OUTPUT < sentences

ASSET_DIR holds the pinned config.json, preprocessor_config.json and
tokenizer.json. Needs torch 2.12.0+cpu, transformers, tokenizers and pillow. It
runs transformers' CLIPModel (eager) on the pinned weights with the reference
tokenizer and image processor, and compares each sentence's logits_per_image with
the lines `transformers_clip_demo score` printed, under the producer's tolerance
(atol 1e-5, rtol 1e-4).
"""
import re, sys
import torch
from PIL import Image
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

assets, weights, image_path, engine_out = sys.argv[1:5]
tok = Tokenizer.from_file(f"{assets}/tokenizer.json")
tok.enable_truncation(max_length=16)
tok.enable_padding(length=16, pad_id=49407, pad_token="<|endoftext|>")
cfg = CLIPConfig.from_json_file(f"{assets}/config.json")
cfg._attn_implementation = "eager"
m = CLIPModel(cfg)
print(m.load_state_dict(load_file(weights), strict=False))
m.eval()
pixels = torch.tensor(
    CLIPImageProcessor.from_pretrained(assets)(images=Image.open(image_path).convert("RGB"), return_tensors="np")["pixel_values"]
)
engine = {}
for line in open(engine_out):
    mm = re.match(r'(".*")\s+logits_per_image = (-?[0-9.]+)', line)
    if mm:
        engine[eval(mm.group(1))] = float(mm.group(2))
atol, rtol = 1e-5, 1e-4
for s in sys.stdin.read().split("\n")[:-1]:
    e = tok.encode(s)
    with torch.no_grad():
        r = m(input_ids=torch.tensor([e.ids]), attention_mask=torch.tensor([e.attention_mask]), pixel_values=pixels)
    ref = r.logits_per_image.item()
    got = engine[s]
    ok = abs(got - ref) <= atol + rtol * abs(ref)
    print(f"{s!r}: engine {got:.6f}, reference {ref:.6f}, diff {abs(got-ref):.2g}, within tolerance: {ok}")
