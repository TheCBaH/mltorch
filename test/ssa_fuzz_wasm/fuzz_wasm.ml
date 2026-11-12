open Ssa_fuzz.Fuzz

let backend =
  let run case =
    let p = case.program in
    match Ssa_wasm.lower p with
    | Error e -> Refused (Fmt.str "%a" Ssa_wasm.pp_error e)
    | Ok lowered -> (
        let loop, bind = emitted case (Ssa_wasm.arguments p) in
        match Err.payload (Loop_wasm_exec.exec_module ~lowered loop ~bind) with
        | Error (#Loop_ir.Loop_interp.error as e) -> of_row e
        | Error e -> Fmt.failwith "%a" Loop_wasm_exec.pp_error e
        | Ok outputs -> of_outputs outputs)
  in
  { name = "wasm"; run }
