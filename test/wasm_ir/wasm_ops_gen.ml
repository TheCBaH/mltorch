(* One exported function per numeric operation, named by its mnemonic. *)
open Wasm

let () =
  (* A vector operation cannot take or return a [v128] across the JS boundary:
     its wrapper takes a memory address per vector operand and a destination
     address, reads the operands with [v128.load] and stores the result. A
     scalar operand (a splat's, a shift count) is passed as is. *)
  let wrapper op =
    let params, results = Wasm_op.signature op in
    let is_vec t = Wasm_type.equal t Wasm_type.V128 in
    if List.exists is_vec params || List.exists is_vec results then
      let wparams =
        List.map (fun t -> if is_vec t then Wasm_type.I32 else t) params
        @ if List.exists is_vec results then [ Wasm_type.I32 ] else []
      in
      let load i t =
        if is_vec t then
          [
            Instr.Local_get i;
            Instr.Simd_load (Simd_load.Load, { Mem_arg.align = 0; offset = 0 });
          ]
        else [ Instr.Local_get i ]
      in
      let dest = List.length params in
      let operands = List.concat (List.mapi load params) in
      let body =
        if List.exists is_vec results then
          [ Instr.Local_get dest ] @ operands @ [ Instr.Numeric op ]
          @ [
              Instr.Simd_store
                (Simd_store.Store, { Mem_arg.align = 0; offset = 0 }, 0);
            ]
        else operands @ [ Instr.Numeric op ]
      in
      {
        Func.type_ =
          {
            Func_type.params = wparams;
            results = List.filter (fun t -> not (is_vec t)) results;
          };
        locals = [];
        body;
      }
    else
      {
        Func.type_ = { Func_type.params; results };
        locals = [];
        body =
          List.mapi (fun i _ -> Instr.Local_get i) params @ [ Instr.Numeric op ];
      }
  in
  (* A relaxed operation needs a flag a default node lacks, so it would make the
     whole module invalid there: it is exercised on its own (relaxed.t). *)
  let ops =
    List.filter
      (fun op -> Wasm_features.of_op op <> Some Wasm_features.Relaxed_simd)
      Wasm_op.all
  in
  let funcs = List.map wrapper ops in
  let exports =
    { Export.name = "memory"; kind = Export.Memory }
    :: List.mapi
         (fun i op -> { Export.name = Wasm_op.name op; kind = Export.Func i })
         ops
  in
  let m =
    {
      Module.empty with
      funcs;
      exports;
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
