(* [Repeat]'s own Const-SSA tests, split out of const_ssa_test.ml once that
   file crossed the tracked 1000-line cap (scripts/check-file-size.sh).
   Shared fixtures live in const_ssa_helpers.ml. *)

open Graph_ir
open Const_ssa_helpers

let%expect_test "Const-SSA: captured input and repeat export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:6) in
  let plan =
    match
      Const_ssa.add Const_ssa.empty ~id:(value 1)
        (Const_ssa.Leaf
           {
             leaf = Const_ssa.Captured (Const_ssa.Capture.of_string "weight");
             output = input;
           })
    with
    | Ok plan ->
        Const_ssa.add plan ~id:(value 2)
          (Const_ssa.Apply
             {
               op =
                 Graph_ir.Repeat
                   {
                     Repeat.Repeat.params =
                       { repeats = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 };
                     x = t_ 1;
                   };
               output;
             })
    | Error e -> Error e
  in
  (match plan with
  | Error e -> Format.printf "%a@." Const_ssa.pp_error (Err.Error.kind e)
  | Ok plan ->
      pp_result Const_ssa.pp_error (Const_ssa.validate plan);
      Format.printf "%a@." Const_ssa.pp plan);
  [%expect
    {|
    ok
    t1 = captured "weight"
    t2 = repeat x=t1 params={repeats=[C=3]} |}]

let%expect_test "Const-SSA: repeat grounds to the wrapped source coordinate" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:6) in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:input
      (Const_ssa.Capture.of_string "w")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Repeat
         {
           Repeat.Repeat.params =
             { repeats = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 };
           x = t_ 1;
         })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  Vec6.iter output.shape (fun coord ->
      match Const_ssa_symbolic.ground arena store (t_ 2) coord with
      | None -> Format.printf "%a -> none@." Vec6.pp_coord coord
      | Some expr ->
          Format.printf "%a -> %a@." Vec6.pp_coord coord Ground_expr.pp expr);
  [%expect
    {|
    (0) -> capture."w"(0)
    (1) -> capture."w"(1)
    (2) -> capture."w"(0)
    (3) -> capture."w"(1)
    (4) -> capture."w"(0)
    (5) -> capture."w"(1) |}]

let%expect_test "Const-SSA: materializes a captured repeat once" =
  let ramp_c shape =
    Tensor.materialize shape (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:6) in
  let store =
    match
      Constant_store.bind_captured Constant_store.empty ~tensor:source
        (Const_ssa.Capture.of_string "w")
    with
    | Error e ->
        failwith
          (Format.asprintf "%a" Constant_store.pp_error (Err.Error.kind e))
    | Ok store ->
        Constant_store.bind_apply store ~tensor:output
          (Graph_ir.Repeat
             {
               Repeat.Repeat.params =
                 { repeats = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 };
               x = t_ 1;
             })
        |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  let resolver = function
    | capture
      when Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "w") ->
        Err.return (ramp_c source.shape)
    | capture -> Err.fail (`Missing_capture capture)
  in
  match Const_ssa_materialize.materialize resolver store with
  | Error e ->
      Format.printf "%a@." Const_ssa_materialize.pp_error (Err.Error.kind e)
  | Ok (store, report) ->
      Format.printf "captures=%d applies=%d cache_hits=%d@." report.captures
        report.applies report.cache_hits;
      Format.printf "%a@." Tensor.pp
        (Tensor_id.Map.find (t_ 2) (Constant_store.materialized store));
      [%expect
        {|
    captures=1 applies=1 cache_hits=1
    tensor f32 [C=6] {0, 1, 0, 1, 0, 1} |}]
