Binding-generator coverage over the full native_functions.yaml. The generator
is gated to a controlled type set; this records how many ops it can currently
emit and why the rest are skipped (the skip buckets are the roadmap for which
types/return-shapes to support next).

  $ ./aten_gen_coverage.exe native_functions.yaml
  total: 2584
  generated: 1165
  skipped: 1419
  top skip reasons:
     631  out= variant
     478  unsupported return shape
      91  unsupported arg type: Layout?
      38  unsupported arg type: float?
      37  unsupported arg type: Generator?
      34  unsupported arg type: str
      31  unsupported arg type: str?
      24  unsupported arg type: Tensor[]
      15  unsupported arg type: float[]?
      14  unsupported arg type: Scalar?
       8  unsupported arg type: Tensor?[]
       6  unsupported arg type: MemoryFormat?
