module R = Machine_rivet_riscv32
module Kernel_refusal = Coralnpu_kernels.Kernel_refusal

let of_items ?(zvt = true) ~name items =
  Kernel_refusal.generate @@ fun esc ->
  let encoder pp e = Kernel_refusal.refuse esc (Encoder (Fmt.str "%a" pp e)) in
  let extensions =
    if zvt then
      match Peak_zvt.zvt with
      | Ok ext -> [ ext ]
      | Error _ -> Kernel_refusal.refuse esc (Encoder "no Zvt encoder")
    else []
  in
  let k =
    match Err.payload (R.Rv32_kernel.finalize ~name items) with
    | Ok k -> k
    | Error e -> encoder R.Rv32_kernel.pp_error e
  in
  match
    Err.payload
      (R.Rv32_kernel_lower.module_ ~extensions
         ~profile:R.Rv32_profile.Rv32imf_zve32f_ilp32f k)
  with
  | Ok m -> m
  | Error e -> encoder R.Rv32_kernel_lower.pp_error e
