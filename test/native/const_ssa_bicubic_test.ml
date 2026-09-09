(* [Upsample_bicubic2d]'s own Const-SSA tests. Its own file (rather than
   folded into const_ssa_test.ml, which is already at the tracked 1000-line
   cap): unlike every other admitted op, its ground form is a real weighted
   blend of up to sixteen captured cells rather than a pure reindexing or a
   one/two-operand arithmetic expression, so it earns a dedicated home the
   same way [Concat]/[Repeat] did. Shared fixtures live in
   const_ssa_helpers.ml. *)

open Graph_ir
open Const_ssa_helpers

let row c = Dim.to_int (Vec6.get c Axis.H)
let col c = Dim.to_int (Vec6.get c Axis.W)

let bicubic_params : Resize.Bicubic2d.params =
  {
    output_size =
      Op_config.Hw.{ h = Op_config.Pos.of_int 3; w = Op_config.Pos.of_int 3 };
    align_corners = false;
  }

let%expect_test "Const-SSA: captured input and upsample_bicubic2d export" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1) in
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
                 Graph_ir.Upsample_bicubic2d
                   { Resize.Bicubic2d.params = bicubic_params; x = t_ 1 };
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
    t2 = upsample_bicubic2d
           x=t1
           params={output_size={h=3; w=3};
           align_corners=false} |}]

(* [Literal], not [Captured]: [Ground_expr.eval] resolves a captured cell
   only through an explicit [Valuation], and this test wants a plain number
   out, the same reason [sigmoid_expr]'s own "grounds to..." test above
   evaluates a literal rather than a capture. The corner coordinate (0,0):
   its real source coordinate is negative on both axes ([Bicubic_axis]'s own
   module doc), so every one of its four TAPS on each axis clamps -- some of
   them to the SAME edge element -- which is exactly the case a
   coordinate-transform bug (e.g. clamping the real coordinate itself, the
   way [Bilinear_axis] does) would get wrong without necessarily crashing.
   Evaluated, not just printed structurally: the expression at this one
   coordinate already has sixteen leaf reads and is not worth eyeballing by
   shape alone. *)
let%expect_test
    "Const-SSA: upsample_bicubic2d grounds to the same blend Compute.pixel \
     produces, at the clamped corner" =
  let input = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1) in
  let literal =
    Tensor.materialize input.shape (fun c -> float_of_int ((2 * row c) + col c))
  in
  let store =
    Constant_store.bind_literal Constant_store.empty ~tensor:input literal
    |> Err.or_raise ~pp_error:Constant_store.pp_error
    |> fun store ->
    Constant_store.bind_apply store ~tensor:output
      (Graph_ir.Upsample_bicubic2d
         { Resize.Bicubic2d.params = bicubic_params; x = t_ 1 })
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  (match Const_ssa_symbolic.ground arena store (t_ 2) Vec6.origin with
  | None -> Format.printf "none@."
  | Some expr ->
      Format.printf "%.6f@." (Ground_expr.eval expr Ground_expr.Valuation.empty));
  [%expect {| -0.260417 |}]

let%expect_test "Const-SSA: materializes a captured upsample_bicubic2d once" =
  let source = sig_ 1 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1) in
  let output = sig_ 2 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1) in
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
          (Graph_ir.Upsample_bicubic2d
             { Resize.Bicubic2d.params = bicubic_params; x = t_ 1 })
        |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  let resolver = function
    | capture
      when Const_ssa.Capture.equal capture (Const_ssa.Capture.of_string "w") ->
        Err.return
          (Tensor.materialize source.shape (fun c ->
               float_of_int ((2 * row c) + col c)))
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
    tensor f32 [H=3 W=3 C=1] {-0.260417, 0.326389, 0.913194, 0.913194, 1.5, 2.08681, 2.08681, 2.67361, ...} |}]
