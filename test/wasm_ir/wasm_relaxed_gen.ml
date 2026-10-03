(* One exported function per relaxed operation, named by its mnemonic, like
   wasm_ops_gen but only for the operations a default node rejects. *)
open Wasm

let () =
  let ops =
    List.filter
      (fun op -> Wasm_features.of_op op = Some Wasm_features.Relaxed_simd)
      Wasm_op.all
  in
  let wrapper op =
    let params, _ = Wasm_op.signature op in
    let n = List.length params in
    {
      Func.type_ =
        {
          Func_type.params =
            List.map (fun _ -> Wasm_type.I32) params @ [ Wasm_type.I32 ];
          results = [];
        };
      locals = [];
      body =
        [ Instr.Local_get n ]
        @ List.concat
            (List.mapi
               (fun i _ ->
                 [
                   Instr.Local_get i;
                   Instr.Simd_load
                     (Simd_load.Load, { Mem_arg.align = 0; offset = 0 });
                 ])
               params)
        @ [
            Instr.Numeric op;
            Instr.Simd_store
              (Simd_store.Store, { Mem_arg.align = 0; offset = 0 }, 0);
          ];
    }
  in
  let m =
    {
      Module.empty with
      funcs = List.map wrapper ops;
      exports =
        { Export.name = "memory"; kind = Export.Memory }
        :: List.mapi
             (fun i op ->
               { Export.name = Wasm_op.name op; kind = Export.Func i })
             ops;
      memory = Some { Memory.min_pages = 1; max_pages = None };
    }
  in
  match Err.payload (Wasm_encode.module_ m) with
  | Ok bytes ->
      let oc = open_out_bin Sys.argv.(1) in
      output_string oc bytes;
      close_out oc
  | Error e ->
      Fmt.epr "%a@." Wasm_check.pp_error e;
      exit 1
