let register () =
  Inference_compare_pt2.Provider.register_unbuilt
    ~names:
      [
        "rivet";
        "rivet-fp32";
        "rivet-fp32s";
        "rivet-scanned";
        "rivet-scanned-fp32";
        "rivet-scanned-fp32s";
        "rivet-table";
        "rivet-table-fp32";
        "rivet-table-fp32s";
        "rivet-table-scanned";
        "rivet-table-scanned-fp32";
        "rivet-table-scanned-fp32s";
      ]
    "the Rivet AArch64 libraries (rivet.aarch64_encode, rivet.driver_direct) \
     are not installed"
