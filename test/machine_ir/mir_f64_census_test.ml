open Machine_ir
open Mir_fixtures

(* The binary64 census: each role is counted by a program whose count is known
   by construction, and a binary32 program counts zero. *)

let census p =
  match Err.payload (Mir_verify.generic p) with
  | Ok g -> Fmt.str "%a" Mir_f64_census.pp (Mir_f64_census.of_program g)
  | Error d -> Fmt.str "%a" Mir_diagnostic.pp d

let helper_id = Mir_id.Helper.of_int 0

let exp_helper ty =
  {
    Mir_helper.id = helper_id;
    name = "exp";
    version = 1;
    params = [ ty ];
    results = [ ty ];
    effects = Mir_helper.Effect.Pure;
    failures = [];
  }

(* [x + x] and a call of [exp x], at one float type. *)
let add_and_call ty =
  let bld = B.create () in
  let entry = B.new_block bld [ ty ] in
  let x = List.hd (B.param entry) in
  let sum = B.emit bld entry (Mir_op.Fbinary (Mir_op.Fbinary.Add, x, x)) in
  let signature = function
    | Mir_op.Callee.Helper _ ->
        Some { Mir_typing.Signature.params = [ ty ]; results = [ ty ] }
    | Mir_op.Callee.Func _ -> None
  in
  let e =
    match
      B.op bld entry ~signature
        (Mir_op.Call (Mir_op.Callee.Helper helper_id, [ sum ]))
    with
    | Ok [ e ] -> e
    | Ok _ | Error _ -> failwith "call"
  in
  B.return entry [ e ];
  program
    ~helpers:[ exp_helper ty ]
    (B.func bld ~id:fn ~name:"f" ~entry ~results:[ ty ])

let%expect_test "known counts" =
  print_endline (census (sum_f32 ()));
  print_endline (census (add_and_call Mir_type.F64));
  [%expect
    {|
    arithmetic 1, conversion 1, helper_call 0, memory_move 1
    arithmetic 1, conversion 0, helper_call 1 (exp x1), memory_move 0 |}]

(* The check can fail: the same program at binary32 counts nothing, so a single
   binary64 add is what turns a zero expectation red. [store_f32] narrows a
   binary64 argument, which the census counts as a conversion. *)
let%expect_test "binary32 counts zero" =
  print_endline (census (add_and_call Mir_type.F32));
  print_endline (census (store_f32 ()));
  [%expect
    {|
    arithmetic 0, conversion 0, helper_call 0, memory_move 0
    arithmetic 0, conversion 1, helper_call 0, memory_move 0 |}]
