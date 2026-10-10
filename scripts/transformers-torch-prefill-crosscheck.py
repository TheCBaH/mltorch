#!/usr/bin/env python3
"""How far is the reference stack from the published outputs?

Runs transformers' LlamaForCausalLM (eager attention) on the pinned SmolLM2
weights for the published prefill cases and counts, per output, the elements
that exceed the producer's tolerance (atol 1e-5, rtol 1e-4) against the
PUBLISHED outputs. Needs torch 2.12.0+cpu (the producer's build) and the paths
below edited to the cache: it is a manual measurement, not a gate. The result
(see the design record) is that this stack, on the producer's architecture, fails
the producer's own tolerance on the same kind of elements the native engine does.
"""
import torch, json, sys
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file
B="/tmp/tf-s0/cache/bundles/51b546590e7ad53065605ac67edfa668bd91597fd9b4620ba6afb835d0c246b7"
cfg=LlamaConfig.from_json_file("/tmp/tf-s0/smolcfg/config.json")
cfg._attn_implementation="eager"
m=LlamaForCausalLM(cfg)
sd=load_file("/tmp/tf-s0/cache/blobs/5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c")
r0=m.load_state_dict(sd,strict=False); print("missing",r0.missing_keys[:3],"unexpected",r0.unexpected_keys[:3])
m.tie_weights(); m.eval()
atol,rtol=1e-5,1e-4
for case in ("case-00","case-01"):
    inp=torch.load(f"{B}/cases/{case}/inputs.pt"); out=torch.load(f"{B}/cases/{case}/outputs.pt")
    with torch.no_grad():
        r=m(input_ids=inp["input_ids"],attention_mask=inp["attention_mask"],use_cache=True)
    mine={"logits":r.logits}
    pkv=r.past_key_values
    for i in range(cfg.num_hidden_layers):
        try:
            k,v=pkv.layers[i].keys,pkv.layers[i].values
        except Exception:
            k,v=pkv[i]
        mine[f"present_{i}_key"]=k; mine[f"present_{i}_value"]=v
    bad_total=0
    for name,t in mine.items():
        ref=out[name]
        diff=(t-ref).abs(); allow=atol+rtol*ref.abs()
        bad=(diff>allow).sum().item(); bad_total+=bad
        if name=="logits" or bad: print(case,name,"bitwise",torch.equal(t,ref),"maxabs",diff.max().item(),"over tolerance",bad,"of",t.numel())
    print(case,"total over tolerance:",bad_total)
