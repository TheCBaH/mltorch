(* A module using every structural form: an import, a loop, if/else with
   a result, a lazy branch, memory loads/stores, a data segment, a mutable
   global and exports of each kind. *)
open Wasm
open Wasm_type

let i32 n = Instr.I32_const n
let op o = Instr.Numeric o
let arg align offset = { Mem_arg.align; offset }

let m =
  let twice = { Func_type.params = [ F64 ]; results = [ F64 ] } in
  {
    Module.imports =
      [ { Import.module_name = "m"; name = "twice"; type_ = twice } ];
    memory = Some { Memory.min_pages = 1; max_pages = None };
    globals = [ { Global.type_ = I32; mutable_ = true; init = i32 5l } ];
    funcs =
      [
        (* sum_f64(ptr, n) -> f64 : sum of n doubles; twice(sum) *)
        {
          Func.type_ = { Func_type.params = [ I32; I32 ]; results = [ F64 ] };
          locals = [ F64; I32 ];
          body =
            [
              Instr.Block
                ( None,
                  [
                    Instr.Loop
                      ( None,
                        [
                          Instr.Local_get 3;
                          Instr.Local_get 1;
                          op Wasm_op.I32_ge_s;
                          Instr.Br_if 1;
                          Instr.Local_get 2;
                          Instr.Local_get 0;
                          Instr.Local_get 3;
                          i32 3l;
                          op Wasm_op.I32_shl;
                          op Wasm_op.I32_add;
                          Instr.Load (Load.F64_load, arg 3 0);
                          op Wasm_op.F64_add;
                          Instr.Local_set 2;
                          Instr.Local_get 3;
                          i32 1l;
                          op Wasm_op.I32_add;
                          Instr.Local_set 3;
                          Instr.Br 0;
                        ] );
                  ] );
              Instr.Local_get 2;
              Instr.Call 0;
            ];
        };
        (* fact(n : i64) -> i64, via a lazy if/else with a result *)
        {
          Func.type_ = { Func_type.params = [ I64 ]; results = [ I64 ] };
          locals = [ I64; I64 ];
          body =
            [
              Instr.I64_const 1L;
              Instr.Local_set 1;
              Instr.Block
                ( None,
                  [
                    Instr.Loop
                      ( None,
                        [
                          Instr.Local_get 0;
                          op Wasm_op.I64_eqz;
                          Instr.Br_if 1;
                          Instr.Local_get 1;
                          Instr.Local_get 0;
                          op Wasm_op.I64_mul;
                          Instr.Local_set 1;
                          Instr.Local_get 0;
                          Instr.I64_const 1L;
                          op Wasm_op.I64_sub;
                          Instr.Local_set 0;
                          Instr.Br 0;
                        ] );
                  ] );
              Instr.Local_get 1;
            ];
        };
        (* pick(c, p) : c ? load8_u(p) : -1, the load only when c is nonzero *)
        {
          Func.type_ = { Func_type.params = [ I32; I32 ]; results = [ I32 ] };
          locals = [];
          body =
            [
              Instr.Local_get 0;
              Instr.If
                ( Some I32,
                  [ Instr.Local_get 1; Instr.Load (Load.I32_load8_u, arg 0 0) ],
                  [ i32 (-1l) ] );
            ];
        };
        (* bump() -> i32 : global += 1 *)
        {
          Func.type_ = { Func_type.params = []; results = [ I32 ] };
          locals = [];
          body =
            [
              Instr.Global_get 0;
              i32 1l;
              op Wasm_op.I32_add;
              Instr.Global_set 0;
              Instr.Global_get 0;
            ];
        };
        (* put_f32(p, x) : store the F32 rounding of x, return it widened *)
        {
          Func.type_ = { Func_type.params = [ I32; F64 ]; results = [ F64 ] };
          locals = [];
          body =
            [
              Instr.Local_get 0;
              Instr.Local_get 1;
              op Wasm_op.F32_demote_f64;
              Instr.Store (Store.F32_store, arg 2 0);
              Instr.Local_get 0;
              Instr.Load (Load.F32_load, arg 2 0);
              op Wasm_op.F64_promote_f32;
            ];
        };
        (* fill_copy(p) : fill 4 bytes with 7, copy them 8 bytes up, read back *)
        {
          Func.type_ = { Func_type.params = [ I32 ]; results = [ I32 ] };
          locals = [];
          body =
            [
              Instr.Local_get 0;
              i32 7l;
              i32 4l;
              Instr.Memory_fill;
              Instr.Local_get 0;
              i32 8l;
              op Wasm_op.I32_add;
              Instr.Local_get 0;
              i32 4l;
              Instr.Memory_copy;
              Instr.Local_get 0;
              Instr.Load (Load.I32_load, arg 2 8);
            ];
        };
      ];
    exports =
      [
        { Export.name = "memory"; kind = Export.Memory };
        { Export.name = "sum_f64"; kind = Export.Func 1 };
        { Export.name = "fact"; kind = Export.Func 2 };
        { Export.name = "pick"; kind = Export.Func 3 };
        { Export.name = "bump"; kind = Export.Func 4 };
        { Export.name = "put_f32"; kind = Export.Func 5 };
        { Export.name = "counter"; kind = Export.Global 0 };
        { Export.name = "fill_copy"; kind = Export.Func 6 };
      ];
    data = [ { Data.offset = 16; bytes = "AB" } ];
    customs = [ { Custom.name = "abi"; payload = "loop-wasm/1" } ];
  }

let () =
  match Err.payload (Wasm_encode.module_ m) with
  | Error e ->
      Fmt.epr "%a@." Wasm_check.pp_error e;
      exit 1
  | Ok bytes ->
      let oc = open_out_bin Sys.argv.(1) in
      output_string oc bytes;
      close_out oc;
      Fmt.pr "%s@." (Wasm_wat.to_string m)
