Binding-generator coverage over the full native_functions.yaml. The generator
is gated to a controlled type set; this records how many ops it can currently
emit and why the rest are skipped (the skip buckets are the roadmap for which
types/return-shapes to support next).

  $ ./aten_gen_coverage.exe native_functions.yaml
  total: 2650
  generated: 1077
  skipped: 1573
  top skip reasons:
     656  out= variant
     477  unsupported return shape
     115  unsupported arg type: ScalarType?
      35  unsupported arg type: float?
      35  unsupported arg type: int?
      33  unsupported arg type: Dimname
      33  unsupported arg type: Generator?
      29  unsupported arg type: str
      25  unsupported arg type: Tensor[]
      17  unsupported arg type: SymInt?
      15  unsupported arg type: SymInt[]?
      14  unsupported arg type: Scalar?
