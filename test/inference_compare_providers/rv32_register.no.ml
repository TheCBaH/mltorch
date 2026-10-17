let register () =
  Inference_compare_pt2.Provider.register_unbuilt
    ~names:
      [
        "compcert-coral-core-mini-highmem-axi";
        "compcert-coral-core-mini-highmem-axi-fp32";
        "compcert-coral-mpact-v2";
        "compcert-coral-mpact-v2-fp32";
        "compcert-rv32";
        "compcert-rv32-fp32";
      ]
    "the embedded riscv32 CompCert is not built (make compcert.rv32.install)"
