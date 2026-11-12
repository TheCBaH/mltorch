open Ssa_fuzz.Fuzz

let backend =
  let run case =
    let p = case.program in
    match Ssa_c.kernel ~name:Loop_c_exec.kernel_name p with
    | Error e -> Refused (Fmt.str "%a" Ssa_c.pp_error e)
    | Ok (kernel, sites) -> (
        let loop, bind = emitted case (Ssa_c.arguments p) in
        match
          Err.payload (Loop_c_exec.exec_kernel ~kernel ~sites loop ~bind)
        with
        | Error (#Loop_ir.Loop_interp.error as e) -> of_row e
        | Error e -> Fmt.failwith "%a" Loop_c_exec.pp_error e
        | Ok outputs -> of_outputs outputs)
  in
  { name = "c"; run }
