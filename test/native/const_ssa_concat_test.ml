(* [Concat]'s own Const-SSA tests, split out of const_ssa_test.ml once that
   file crossed the tracked 1000-line cap (scripts/check-file-size.sh).
   Shared fixtures live in const_ssa_helpers.ml. *)

open Graph_ir
open Const_ssa_helpers

let%expect_test "Const-SSA: captured inputs and concat export" =
  let a = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let b = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) in
  let output = sig_ 3 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:5) in
  let plan =
    let open Err.Syntax in
    let* plan =
      Const_ssa.add Const_ssa.empty ~id:(value 1)
        (Const_ssa.Leaf
           {
             leaf = Const_ssa.Captured (Const_ssa.Capture.of_string "a");
             output = a;
           })
    in
    let* plan =
      Const_ssa.add plan ~id:(value 2)
        (Const_ssa.Leaf
           {
             leaf = Const_ssa.Captured (Const_ssa.Capture.of_string "b");
             output = b;
           })
    in
    Const_ssa.add plan ~id:(value 3)
      (Const_ssa.Apply
         {
           op =
             Graph_ir.Concat
               { Concat.Concat.params = { axis = Axis.C }; xs = [ t_ 1; t_ 2 ] };
           output;
         })
  in
  (match plan with
  | Error e -> Format.printf "%a@." Const_ssa.pp_error (Err.Error.kind e)
  | Ok plan ->
      pp_result Const_ssa.pp_error (Const_ssa.validate plan);
      Format.printf "%a@." Const_ssa.pp plan);
  [%expect
    {|
    ok
    t1 = captured "a"
    t2 = captured "b"
    t3 = concat xs=[t1, t2] params={axis=C} |}]

let%expect_test
    "Const-SSA: concat grounds to the offset operand's own coordinate" =
  let a = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let b = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) in
  let output = sig_ 3 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:5) in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:a
      (Const_ssa.Capture.of_string "a")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_captured store ~tensor:b
      (Const_ssa.Capture.of_string "b")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Concat
         { Concat.Concat.params = { axis = Axis.C }; xs = [ t_ 1; t_ 2 ] })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  Vec6.iter output.shape (fun coord ->
      match Const_ssa_symbolic.ground arena store (t_ 3) coord with
      | None -> Format.printf "%a -> none@." Vec6.pp_coord coord
      | Some expr ->
          Format.printf "%a -> %a@." Vec6.pp_coord coord Ground_expr.pp expr);
  [%expect
    {|
    (0) -> capture."a"(0)
    (1) -> capture."a"(1)
    (2) -> capture."b"(0)
    (3) -> capture."b"(1)
    (4) -> capture."b"(2) |}]

let%expect_test "Const-SSA: materializes a captured concat once" =
  let ramp_c shape =
    Tensor.materialize shape (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  let a = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) in
  let b = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) in
  let output = sig_ 3 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:5) in
  let store =
    match
      Constant_store.bind_captured Constant_store.empty ~tensor:a
        (Const_ssa.Capture.of_string "a")
    with
    | Error e ->
        failwith
          (Format.asprintf "%a" Constant_store.pp_error (Err.Error.kind e))
    | Ok store -> (
        match
          Constant_store.bind_captured store ~tensor:b
            (Const_ssa.Capture.of_string "b")
        with
        | Error e ->
            failwith
              (Format.asprintf "%a" Constant_store.pp_error (Err.Error.kind e))
        | Ok store ->
            Constant_store.bind_apply store ~tensor:output
              (Graph_ir.Concat
                 {
                   Concat.Concat.params = { axis = Axis.C };
                   xs = [ t_ 1; t_ 2 ];
                 })
            |> Err.or_raise ~pp_error:Constant_store.pp_error)
  in
  let resolver = function
    | capture
      when Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "a") ->
        Err.return (ramp_c a.shape)
    | capture
      when Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "b") ->
        Err.return (ramp_c b.shape)
    | capture -> Err.fail (`Missing_capture capture)
  in
  match Const_ssa_materialize.materialize resolver store with
  | Error e ->
      Format.printf "%a@." Const_ssa_materialize.pp_error (Err.Error.kind e)
  | Ok (store, report) ->
      Format.printf "captures=%d applies=%d cache_hits=%d@." report.captures
        report.applies report.cache_hits;
      Format.printf "%a@." Tensor.pp
        (Tensor_id.Map.find (t_ 3) (Constant_store.materialized store));
      [%expect
        {|
    captures=2 applies=1 cache_hits=2
    tensor f32 [C=5] {0, 1, 0, 1, 2} |}]
