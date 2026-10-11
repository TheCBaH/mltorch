let register () =
  Inference_compare_pt2.Provider.register_unbuilt ~names:[ "compcert" ]
    "the embedded CompCert is not built (MLTORCH_COMPCERT=<isa> with the \
     rivet_compcert package installed)"
