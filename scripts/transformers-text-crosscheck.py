#!/usr/bin/env python3
"""Check the BERT-tiny text example against the reference stack.

Usage: transformers-text-crosscheck.py VOCAB.txt CONFIG.json SAFETENSORS DUMP.jsonl < sentences

Needs torch and `tokenizers` (the producer's torch build: 2.12.0+cpu, aarch64).
For each input line it tokenizes with the Rust `tokenizers` WordPiece (the
backend of BertTokenizerFast), runs transformers' BertModel in eager mode on the
pinned weights, and compares last_hidden_state and pooler_output with the
engine's dump (`transformers_text_demo --dump`) under the producer's
tolerances (atol 1e-5, rtol 1e-4, torch.testing.assert_close's rule). Prints
the element counts that exceed it. A reference run on another machine is not
the producer's run: see the design record on how far torch itself is from the
published outputs.
"""
import json, sys
import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertModel
from safetensors.torch import load_file

vocab, config, weights, dump = sys.argv[1:5]
tok = BertWordPieceTokenizer(vocab, lowercase=True)
tok.enable_truncation(max_length=16)
tok.enable_padding(length=16, pad_id=0, pad_token="[PAD]")
m = BertModel(BertConfig.from_json_file(config), add_pooling_layer=True)
sd = {k[5:]: v for k, v in load_file(weights).items()
      if k.startswith("bert.") and "position_ids" not in k}
m.load_state_dict(sd, strict=True)
m.eval()
sentences = sys.stdin.read().split("\n")[:-1]
rows = [json.loads(l) for l in open(dump)]
assert len(rows) == len(sentences), (len(rows), len(sentences))
atol, rtol = 1e-5, 1e-4
over = {"hidden": 0, "pooled": 0}
worst = {"hidden": 0.0, "pooled": 0.0}
total = {"hidden": 0, "pooled": 0}
for s, row in zip(sentences, rows):
    e = tok.encode(s)
    ids = torch.tensor([e.ids])
    mask = torch.tensor([e.attention_mask])
    with torch.no_grad():
        r = m(input_ids=ids, attention_mask=mask, token_type_ids=torch.zeros_like(ids))
    for key, ref in (("hidden", r.last_hidden_state.flatten()), ("pooled", r.pooler_output.flatten())):
        ours = torch.tensor(row[key], dtype=torch.float32)
        d = (ours - ref).abs()
        over[key] += (d > atol + rtol * ref.abs()).sum().item()
        worst[key] = max(worst[key], d.max().item())
        total[key] += ref.numel()
for k in over:
    print(f"{k}: {over[k]} of {total[k]} elements over tolerance, worst abs {worst[k]:.3g}")
print(f"{len(sentences)} sentences")
