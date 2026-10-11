let register () =
  Inference_compare_pt2.Provider.register_unbuilt
    ~names:[ "rivet"; "rivet-scanned"; "rivet-table"; "rivet-table-scanned" ]
    "the Rivet AArch64 libraries (rivet.aarch64_encode, rivet.driver_direct) \
     are not installed"
