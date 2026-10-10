#!/usr/bin/env python3
"""Check bin/transformers_generate_demo against transformers' greedy generation.

Usage: transformers-generate-crosscheck.py CONFIG.json SAFETENSORS ID,ID,ID,ID [ENGINE_LOG]

Needs torch 2.12.0+cpu (the producer's build). Runs LlamaForCausalLM (eager
attention) on the pinned SmolLM2 weights, generates two tokens greedily from the
four prompt ids, and prints them with the top logits. With ENGINE_LOG, the
demo's output (run with GENERATE_DUMP_LOGITS=1), also compares the generated
tokens and the full logits of both steps under the producer's tolerance.
"""
import sys
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file

cfg_path, weights, ids = sys.argv[1:4]
log = sys.argv[4] if len(sys.argv) > 4 else None
cfg = LlamaConfig.from_json_file(cfg_path)
cfg._attn_implementation = "eager"
m = LlamaForCausalLM(cfg)
m.load_state_dict(load_file(weights), strict=False)
m.tie_weights()
m.eval()
prompt = torch.tensor([[int(x) for x in ids.split(",")]])
mask = torch.ones_like(prompt)
with torch.no_grad():
    out = m.generate(prompt, attention_mask=mask, max_new_tokens=2, do_sample=False,
                     output_logits=True, return_dict_in_generate=True)
tokens = out.sequences[0].tolist()
print("reference generated:", ",".join(map(str, tokens)))
ref_logits = [l[0] for l in out.logits]  # one [vocab] per step
if log:
    engine_tokens, engine_logits = None, {}
    for line in open(log):
        if line.startswith("generated:"):
            engine_tokens = [int(x) for x in line.split(":")[1].strip().split(",")]
        elif line.startswith("logits "):
            name, rest = line[len("logits "):].split(":", 1)
            engine_logits[name] = torch.tensor([float(x) for x in rest.split()])
    print("engine generated:   ", ",".join(map(str, engine_tokens)))
    print("tokens equal:", engine_tokens == tokens)
    atol, rtol = 1e-5, 1e-4
    for name, ref in zip(("prefill", "decode"), ref_logits):
        d = (engine_logits[name] - ref).abs()
        over = (d > atol + rtol * ref.abs()).sum().item()
        print(f"{name}: max abs {d.max().item():.3g}, {over} of {ref.numel()} over tolerance")
