Binding-generator coverage over the full native_functions.yaml. The generator
is gated to a controlled type set; this records how many ops it can currently
emit and why the rest are skipped (the skip buckets are the roadmap for which
types/return-shapes to support next).

  $ ./aten_gen_coverage.exe native_functions.yaml
  total: 2691
  generated: 1083
  skipped: 1608
  top skip reasons:
     661  out= variant
     495  unsupported return shape
     115  unsupported arg type: ScalarType?
      38  unsupported arg type: Generator?
      38  unsupported arg type: float?
      35  unsupported arg type: int?
      33  unsupported arg type: Dimname
      29  unsupported arg type: str
      27  unsupported arg type: Tensor[]
      17  unsupported arg type: SymInt?
      16  unsupported arg type: SymInt[]?
      14  unsupported arg type: Scalar?
