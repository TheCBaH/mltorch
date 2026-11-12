open Ssa_fuzz.Fuzz

let backend =
  let run case =
    let p = case.program in
    match Ssa_js.program p with
    | Error e -> Refused (Fmt.str "%a" Ssa_js.pp_error e)
    | Ok (ast, sites) -> (
        let loop, bind = emitted case (Ssa_js.arguments p) in
        let compiled =
          match
            Err.payload
              (Loop_js_exec.compile_kernel ~sites loop
                 (Js_print.factory_body ast))
          with
          | Ok c -> c
          | Error (`Js_compile m) ->
              Fmt.failwith "the engine refused the source: %s" m
        in
        match Err.payload (Loop_js_exec.run compiled ~bind) with
        | Error (#Loop_ir.Loop_interp.error as e) -> of_row e
        | Error e -> Fmt.failwith "%a" Loop_js_exec.pp_error e
        | Ok outputs -> of_outputs outputs)
  in
  { name = "js"; run }
