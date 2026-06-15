Binding-generator coverage over the full native_functions.yaml. The generator
is gated to a controlled type set; this records how many ops it can currently
emit and why the rest are skipped (the skip buckets are the roadmap for which
types/return-shapes to support next).

  $ ./aten_gen_coverage.exe native_functions.yaml
  total: 2584
  generated: 1120
  skipped: 1464
  top skip reasons:
     631  out= variant
     478  unsupported return shape
     115  unsupported arg type: ScalarType?
      38  unsupported arg type: float?
      37  unsupported arg type: Generator?
      34  unsupported arg type: str
      24  unsupported arg type: Tensor[]
      19  unsupported arg type: str?
      16  unsupported arg type: SymInt[]?
      13  unsupported arg type: int[1]?
      12  unsupported arg type: Scalar?
      12  unsupported arg type: SymInt[1]?
