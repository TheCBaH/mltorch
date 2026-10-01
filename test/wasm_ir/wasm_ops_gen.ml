(* One exported function per numeric operation, named by its mnemonic. *)
open Wasm

let () =
  let funcs =
    List.map
      (fun op ->
        let params, results = Wasm_op.signature op in
        {
          Func.type_ = { Func_type.params; results };
          locals = [];
          body =
            List.mapi (fun i _ -> Instr.Local_get i) params
            @ [ Instr.Numeric op ];
        })
      Wasm_op.all
  in
  let exports =
    List.mapi
      (fun i op -> { Export.name = Wasm_op.name op; kind = Export.Func i })
      Wasm_op.all
  in
  let m = { Module.empty with funcs; exports } in
  match Err.payload (Wasm_encode.module_ m) with
  | Ok bytes ->
      let oc = open_out_bin Sys.argv.(1) in
      output_string oc bytes;
      close_out oc
  | Error e ->
      Fmt.epr "%a@." Wasm_check.pp_error e;
      exit 1
