Binding-generator coverage over the full native_functions.yaml. The generator
is gated to a controlled type set; this records how many ops it can currently
emit and why the rest are skipped (the skip buckets are the roadmap for which
types/return-shapes to support next).

  $ ./aten_gen_coverage.exe native_functions.yaml
  total: 2584
  generated: 1292
  skipped: 1292
  top skip reasons:
     631  out= variant
     301  unsupported return shape
      90  unsupported arg type: Device?
      47  unsupported arg type: float?
      39  unsupported arg type: str
      38  unsupported arg type: Generator?
      34  unsupported arg type: Tensor[]
      33  unsupported arg type: str?
      16  unsupported arg type: float[]?
      16  unsupported return shape: 4-tuple
      12  unsupported arg type: bool[3]
       9  unsupported return shape: 5-tuple
