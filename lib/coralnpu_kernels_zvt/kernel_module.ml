module R = Machine_rivet_riscv32

let of_items ?(zvt = true) ~name items =
  let ( let* ) = Result.bind in
  let* extensions =
    if zvt then
      match Peak_zvt.zvt with
      | Ok ext -> Ok [ ext ]
      | Error _ -> Error "no Zvt encoder"
    else Ok []
  in
  let* k =
    Result.map_error
      (Fmt.str "%a" R.Rv32_kernel.pp_error)
      (Err.payload (R.Rv32_kernel.finalize ~name items))
  in
  Result.map_error
    (Fmt.str "%a" R.Rv32_kernel_lower.pp_error)
    (Err.payload
       (R.Rv32_kernel_lower.module_ ~extensions
          ~profile:R.Rv32_profile.Rv32imf_zve32f_ilp32f k))
