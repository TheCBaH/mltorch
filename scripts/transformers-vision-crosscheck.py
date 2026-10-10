import sys, subprocess, numpy as np, torch
from PIL import Image
from safetensors.torch import load_file
from transformers import MobileViTConfig, MobileViTForImageClassification, MobileViTImageProcessor
cfg = MobileViTConfig.from_json_file('/tmp/tf-s0/mvit/config.json')
m = MobileViTForImageClassification(cfg)
print(m.load_state_dict(load_file('/tmp/tf-s0/cache/blobs/635a7cdd86f6ffd9e6ceb2f0b5b5f1971af129d12cb693ef70fe2ac471ff6399'), strict=False))
m.eval()
proc = MobileViTImageProcessor.from_pretrained('/tmp/tf-s0/mvit')
for name in sys.argv[1:]:
    img = Image.open(f'/tmp/tf-s0/{name}.png' if name == 'scene' else f'/tmp/tf-s0/{name}.jpg').convert('RGB')
    ref_pix = proc(images=img, return_tensors='pt')['pixel_values']
    out = subprocess.run(['/workspaces/mltorch/_build/default/bin/transformers_vision_demo.exe', 'pixels', f'/tmp/tf-s0/{name}.ppm'], capture_output=True, text=True, cwd='/workspaces/mltorch').stdout.split()
    ours = torch.tensor([float(x) for x in out]).reshape(1, 3, 256, 256)
    print(name, 'pixels: max abs diff', (ours - ref_pix).abs().max().item(), 'equal shapes', ours.shape == ref_pix.shape)
    with torch.no_grad(): logits = m(pixel_values=ref_pix).logits[0]
    top = logits.topk(5)
    print(' reference top-5:', [(i.item(), round(v.item(), 6)) for v, i in zip(top.values, top.indices)])
    torch.save(logits, f'/tmp/tf-s0/ref_logits_{name}.pt')
