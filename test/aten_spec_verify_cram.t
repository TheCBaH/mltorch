End-to-end check of the bin/aten_spec_verify CLI: a spec read from stdin is
synthesized, run on both the ATen and native paths, and verified. The exhaustive
generator coverage lives in the (faster, in-process) aten_spec_run_test expect
tests; this keeps just the stdin + --print plumbing through the real binary.

verify mode (stdin via "-"), add.Tensor with a different generator per input:

  $ ../bin/aten_spec_verify.exe - <<'JSON'
  > { "target": "torch.ops.aten.add.Tensor",
  >   "args": {
  >     "self":  { "dtype": "f32", "shape": [2, 3], "random": { "uniform": { "low": -1.0, "high": 1.0 } } },
  >     "other": { "dtype": "f32", "shape": [2, 3], "random": { "normal":  { "mean": 0.0, "variance": 1.0 } } }
  >   } }
  > JSON
  [spec] torch.ops.aten.add.Tensor: matched

--print mode: evaluate and print the output tensor from both paths:

  $ ../bin/aten_spec_verify.exe --print <<'JSON'
  > { "target": "torch.ops.aten.add.Tensor",
  >   "args": {
  >     "self":  { "dtype": "f32", "shape": [2], "values": [{"float":"0x3f800000"}, {"float":"0x40000000"}] },
  >     "other": { "dtype": "f32", "shape": [2], "values": [{"float":"0x3f000000"}, {"float":"0x3f000000"}] }
  >   } }
  > JSON
  [eval] torch.ops.aten.add.Tensor
    aten   out0 = [1.5; 2.5]
    native out0 = [1.5; 2.5]
