open Wasm

(* The binary64 census over hand-built modules whose counts are known, and the
   op table's classification against each operation's own signature. *)

let has_f64 s =
  let n = String.length s in
  let rec go i = i + 3 <= n && (String.sub s i 3 = "f64" || go (i + 1)) in
  go 0

(* An operation is classified exactly when its mnemonic says f64, and a scalar
   operation whose signature mentions f64 says so: a new mnemonic cannot slip
   through unclassified. *)
let%expect_test "classification agrees with names and signatures" =
  let bad =
    List.filter
      (fun op ->
        let named = has_f64 (Wasm_op.name op) in
        let classified = Option.is_some (Wasm_census.role_of_op op) in
        let params, results = Wasm_op.signature op in
        let typed = List.mem Wasm_type.F64 (params @ results) in
        named <> classified || (typed && not named))
      Wasm_op.all
  in
  Fmt.pr "%d mismatches: %s@." (List.length bad)
    (String.concat " " (List.map Wasm_op.name bad));
  [%expect {| 0 mismatches: |}]

let func params results locals body =
  { Func.type_ = { Func_type.params; results }; locals; body }

let census m = Fmt.str "%a" Wasm_census.pp (Wasm_census.of_module m)

(* x + x, then exp (imported from [math]) of the sum, at one float type. *)
let add_and_call ty (add : Wasm_op.t) =
  {
    Module.empty with
    imports =
      [
        {
          Import.module_name = "math";
          name = "exp";
          type_ = { Func_type.params = [ ty ]; results = [ ty ] };
        };
      ];
    funcs =
      [
        func [ ty ] [ ty ] []
          [
            Instr.Local_get 0;
            Instr.Local_get 0;
            Instr.Numeric add;
            Instr.Call 0;
          ];
      ];
  }

let%expect_test "known counts" =
  print_endline (census (add_and_call Wasm_type.F64 Wasm_op.F64_add));
  print_endline (census (add_and_call Wasm_type.F32 Wasm_op.F32_add));
  print_endline
    (census
       {
         Module.empty with
         funcs =
           [
             func [ Wasm_type.F32 ] [ Wasm_type.F64 ] []
               [
                 Instr.Local_get 0;
                 Instr.Numeric Wasm_op.F64_promote_f32;
                 Instr.F64_const 0L;
                 Instr.Numeric Wasm_op.F64_mul;
               ];
           ];
       });
  [%expect
    {|
    arithmetic 1, conversion 0, helper_call 1, memory_move 2; imports 1
    arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 1
    arithmetic 1, conversion 1, helper_call 0, memory_move 1; imports 0 |}]

let%expect_test "helpers and imports are named" =
  let c = Wasm_census.of_module (add_and_call Wasm_type.F64 Wasm_op.F64_add) in
  List.iter (fun (n, k) -> Fmt.pr "%s x%d@." n k) (Wasm_census.helpers c);
  print_endline (String.concat " " (Wasm_census.imports c));
  [%expect {|
    math.exp x1
    math.exp |}]
