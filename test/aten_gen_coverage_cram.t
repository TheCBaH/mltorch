Binding-generator coverage over the full native_functions.yaml. The generator
is gated to a controlled type set; this records how many ops it can currently
emit and why the rest are skipped (the skip buckets are the roadmap for which
types/return-shapes to support next).

  $ ./aten_gen_coverage.exe native_functions.yaml
  total: 2584
  generated: 1619
  skipped: 965
  top skip reasons:
     631  out= variant
     185  unsupported return shape
      38  unsupported arg type: Generator?
      33  unsupported arg type: str?
      17  unsupported arg type: float[]?
      16  unsupported return shape: 4-tuple
      14  unsupported arg type: Scalar[]
      12  unsupported arg type: bool[3]
       9  unsupported return shape: 5-tuple
       6  unsupported arg type: bool[2]
       2  unsupported arg type: Storage
       1  unsupported return shape: 6-tuple
