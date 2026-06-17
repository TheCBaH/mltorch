Binding-generator coverage over the full native_functions.yaml. The generator
is gated to a controlled type set; this records how many ops it can currently
emit and why the rest are skipped (the skip buckets are the roadmap for which
types/return-shapes to support next).

  $ ./aten_gen_coverage.exe native_functions.yaml
  total: 2584
  generated: 1360
  skipped: 1224
  top skip reasons:
     631  out= variant
     301  unsupported return shape
      91  unsupported arg type: Device?
      48  unsupported arg type: float?
      38  unsupported arg type: Generator?
      33  unsupported arg type: str?
      16  unsupported arg type: float[]?
      16  unsupported return shape: 4-tuple
      12  unsupported arg type: bool[3]
       9  unsupported arg type: bool?
       9  unsupported return shape: 5-tuple
       8  unsupported arg type: Tensor?[]
